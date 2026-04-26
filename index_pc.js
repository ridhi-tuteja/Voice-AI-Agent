import express from "express";
import cors from "cors";
import fs from "fs";
import csv from "csv-parser";
import axios from "axios";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";
import { v4 as uuidv4 } from "uuid";
import FormData from "form-data";
import multer from "multer";

const sessions = {};
const MAX_HISTORY = 5;

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

// ── STT: Groq Whisper (free, no VPN issues, 7200 sec/day) ────────────────────
async function speechToText(filePath, mimeType) {
  const form = new FormData();
  form.append("file", fs.createReadStream(filePath), {
    filename: "audio.webm",
    contentType: mimeType || "audio/webm",
  });
  form.append("model", "whisper-large-v3-turbo");
  form.append("response_format", "json");
  form.append("language", "en");

  const res = await axios.post(
    "https://api.groq.com/openai/v1/audio/transcriptions",
    form,
    {
      headers: {
        Authorization: "Bearer " + process.env.GROQ_API_KEY,
        ...form.getHeaders(),
      },
      timeout: 30000,
    }
  );
  return res.data.text || "";
}

// ── TTS: Groq PlayAI TTS (free, same GROQ_API_KEY) ───────────────────────────
// Docs: https://console.groq.com/docs/text-to-speech
// Available voices: https://console.groq.com/docs/text-to-speech#supported-voices
async function textToSpeech(text) {
  const res = await axios.post(
    "https://api.groq.com/openai/v1/audio/speech",
    {
       model: "canopylabs/orpheus-v1-english",
       voice: "diana",
       input: text,
       response_format: "wav",
    },
    {
      headers: {
        Authorization: "Bearer " + process.env.GROQ_API_KEY,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      responseType: "arraybuffer",
      timeout: 30000,
      maxBodyLength: Infinity,
    }
  );

  if (!res.data || res.data.byteLength === 0) {
    throw new Error("TTS returned empty audio buffer");
  }

  return Buffer.from(res.data);
}

// ── Embeddings + Gemini ───────────────────────────────────────────────────────

let extractor;
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

async function initializeEmbeddingModel() {
  extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
}

async function generateEmbeddings(text) {
  if (!extractor) await initializeEmbeddingModel();
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// ── Pinecone ──────────────────────────────────────────────────────────────────

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

async function deleteOldVectors() {
  try {
    const stats = await index.describeIndexStats();
    const ids = Object.keys(stats.namespaces?.[""]?.vectorCount ?? {}).map(
      (id) => id.toString()
    );
    if (ids.length > 0) {
      await index.delete({ ids });
      console.log(`Deleted ${ids.length} old vectors.`);
    }
  } catch (error) {
    console.error("Error deleting old vectors:", error.message);
  }
}

async function loadEmbeddings() {
  const hospitals = [];

  await new Promise((resolve, reject) => {
    fs.createReadStream("hospital_data.csv")
      .pipe(csv())
      .on("data", (row) => hospitals.push(row))
      .on("end", resolve)
      .on("error", reject);
  });

  await deleteOldVectors();

  const vectors = [];
  for (let i = 0; i < hospitals.length; i++) {
    const h = hospitals[i];
    const text = `Hospital Name: ${h["HOSPITAL NAME"]}\nAddress: ${h["Address"]}\nCity: ${h["CITY"]}`;
    const embedding = await generateEmbeddings(text);
    vectors.push({
      id: `hospital_${i}`,
      values: embedding,
      metadata: {
        name: h["HOSPITAL NAME"],
        address: h["Address"],
        city: h["CITY"],
        content: text,
      },
    });
  }

  const BATCH_SIZE = 50;
  for (let i = 0; i < vectors.length; i += BATCH_SIZE) {
    await index.upsert(vectors.slice(i, i + BATCH_SIZE));
    console.log(`Upserted batch ${Math.floor(i / BATCH_SIZE) + 1}`);
  }

  console.log("Hospital embeddings stored in Pinecone.");
}

// ── RAG Query ─────────────────────────────────────────────────────────────────

// Expand user query with known spelling variants and aliases
function expandQuery(query) {
  const aliases = {
    "sajapur": "sarjapur",
    "sajjapur": "sarjapur", 
    "sarjapur": "sarjapur",
    "koramangala": "koramangala",
    "bengalore": "bengaluru",
    "bangalore": "bengaluru",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "gurgaon": "gurugram",
    "new delhi": "delhi",
    "dwarka": "dwarka",
    "connaught": "connaught place",
  };
  let expanded = query.toLowerCase();
  for (const [typo, correct] of Object.entries(aliases)) {
    if (expanded.includes(typo)) {
      expanded = expanded.replace(typo, correct);
    }
  }
  return expanded;
}

async function queryPinecone(query) {
  const queryEmbedding = await generateEmbeddings(query);
  return await index.query({
    vector: queryEmbedding,
    topK: 10,
    includeMetadata: true,
  });
}

async function generateResponse(query, history = []) {
  try {
    // Expand query with fuzzy city/area terms for better recall
    const expandedQuery = expandQuery(query);
    console.log("Expanded query:", expandedQuery);

    const results = await queryPinecone(expandedQuery);

    if (!results || results.matches.length === 0) {
      return "Sorry, I could not find any relevant information.";
    }

    // Filter matches by score threshold to avoid irrelevant results
    const goodMatches = results.matches.filter(m => m.score > 0.3);
    const matchesToUse = goodMatches.length > 0 ? goodMatches : results.matches.slice(0, 3);

    const context = matchesToUse
      .map((m) => `Hospital: ${m.metadata.name}\nAddress: ${m.metadata.address}\nCity: ${m.metadata.city}`)
      .join("\n\n");

    const historyText = history
      .map((h) => `${h.role === "user" ? "User" : "Assistant"}: ${h.text}`)
      .join("\n");

    const prompt = `You are Loop AI, a friendly voice assistant for the Loop Health hospital network.
Responses will be spoken aloud — keep them natural and concise (3-5 sentences max).
Do NOT give medical advice. Only use hospital information from the context below.
If the question is unrelated to hospitals, say: "I\'m sorry, I can\'t help with that. I am forwarding this to a human agent."

IMPORTANT RULES:
- Always include the full address of each hospital in your response.
- If user asks for hospitals near an area or locality (like Sarjapur, Dwarka, Connaught Place), 
  look for hospitals whose address contains that area name — not just the city.
- If no hospital address matches the requested area, say so honestly and suggest nearby cities instead.
- Never say "You can find their addresses in the Loop Health network" — always say the address directly.

Conversation so far:
${historyText}

Context (hospitals from database):
${context}

User question: ${query}

Respond naturally as if speaking. Include hospital names AND their full addresses.
If no relevant match found for the specific area, say: "I couldn\'t find hospitals specifically near [area], but here are some in [city]: ..."`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  } catch (error) {
    console.error("Error generating response:", error.message);
    return "Sorry, there was a problem generating the response.";
  }
}

// ── Routes ────────────────────────────────────────────────────────────────────

const upload = multer({ dest: "uploads/" });

app.post("/initialize", async (req, res) => {
  const token = req.headers.authorization;
  if (token !== `Bearer ${process.env.ADMIN_SECRET}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  try {
    await loadEmbeddings();
    res.status(200).send("Pinecone index initialized successfully.");
  } catch (error) {
    res.status(500).send("Initialization failed: " + error.message);
  }
});

app.post("/query", async (req, res) => {
  try {
    let { question, sessionId } = req.body;
    if (!question) return res.status(400).send("Please provide a question.");

    if (!sessionId) {
      sessionId = uuidv4();
      sessions[sessionId] = [];
    }

    const history = sessions[sessionId] || [];
    const response = await generateResponse(question, history);

    history.push({ role: "user", text: question });
    history.push({ role: "assistant", text: response });
    sessions[sessionId] = history.slice(-MAX_HISTORY * 2);

    res.json({ response, sessionId });
  } catch (error) {
    res.status(500).send("Query failed: " + error.message);
  }
});

// ── Voice Route ───────────────────────────────────────────────────────────────

app.post("/voice", upload.single("audio"), async (req, res) => {
  const filePath = req.file && req.file.path;

  try {
    console.log("----- VOICE PIPELINE START -----");

    if (!req.file) {
      return res.status(400).send("No audio uploaded");
    }

    console.log("File received:", req.file.originalname, "| size:", req.file.size);

    // 1. Groq Whisper STT
    console.log("STT (Groq Whisper)...");
    const userText = await speechToText(filePath, req.file.mimetype);
    console.log("Transcript:", userText);

    if (!userText || !userText.trim()) {
      throw new Error("Empty transcript returned");
    }

    // 2. RAG + Gemini
    console.log("RAG + Gemini...");
    let answer = await generateResponse(userText);
    answer = answer.slice(0, 500);
    console.log("Answer:", answer);

    // 3. Groq PlayAI TTS
    console.log("TTS (Groq PlayAI)...");
    let audioBuffer;
    try {
      audioBuffer = await textToSpeech(answer);
    } catch (ttsErr) {
      const detail = ttsErr.response && ttsErr.response.data
        ? Buffer.from(ttsErr.response.data).toString()
        : ttsErr.message;
      console.error("TTS error detail:", ttsErr.response && ttsErr.response.status, detail);
      throw new Error("TTS failed: " + detail);
    }
    console.log("Audio size:", audioBuffer.length);

    // 4. Cleanup + respond
    fs.unlinkSync(filePath);
    console.log("----- VOICE PIPELINE END -----");

    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Content-Length", audioBuffer.length);
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("X-Transcript", encodeURIComponent(userText));
    res.setHeader("X-Response-Text", encodeURIComponent(answer));
    res.end(audioBuffer);

  } catch (err) {
    console.error("VOICE ERROR:", err.message);
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
    res.status(500).send("Voice pipeline failed: " + err.message);
  }
});

// ── Health ────────────────────────────────────────────────────────────────────

app.get("/health", async (req, res) => {
  try {
    await index.describeIndexStats();
    res.status(200).json({ status: "healthy" });
  } catch (error) {
    res.status(500).json({ status: "unhealthy", error: error.message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 4000;
app.listen(PORT, "0.0.0.0", async () => {
  console.log(`Server running on port ${PORT}`);
  if (process.env.AUTO_INITIALIZE === "True") {
    try {
      await loadEmbeddings();
    } catch (err) {
      console.error("Auto-initialization failed:", err);
    }
  }
});