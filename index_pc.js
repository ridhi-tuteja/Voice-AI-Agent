import express from "express";
import cors from "cors";
import fs from "fs";
import csv from "csv-parser";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";
import { v4 as uuidv4 } from "uuid";
import multer from "multer";

dotenv.config();

const app = express();
app.use(express.json());
// app.use(cors());

app.use(cors({
  exposedHeaders: ["session-id"]
}));

const sessions = {};
const MAX_HISTORY = 5;

const INTRO_MESSAGE =
"Hello! I am Voice AI Agent, your hospital network assistant. How can I help you today?";

const upload = multer({ dest: "uploads/" });

/* ------------------ ElevenLabs TTS ------------------ */

async function textToSpeech(text) {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${process.env.ELEVENLABS_VOICE_ID}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": process.env.ELEVENLABS_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        model_id: "eleven_multilingual_v2",
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.7,
        },
      }),
    }
  );

  const buffer = await response.arrayBuffer();
  return Buffer.from(buffer);
}

/* ------------------ Embeddings Setup ------------------ */

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

/* ------------------ Pinecone Setup ------------------ */

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

/* ------------------ Load CSV & Store Embeddings ------------------ */

async function loadEmbeddings() {
  try {
    const hospitals = [];

    fs.createReadStream("hospital_data.csv")
      .pipe(csv())
      .on("data", (row) => {
        hospitals.push(row);
      })
      .on("end", async () => {
        const vectors = [];

        for (let i = 0; i < hospitals.length; i++) {
          const h = hospitals[i];

          const text = `
            Hospital Name: ${h["HOSPITAL NAME"]}
            Address: ${h["Address"]}
            City: ${h["CITY"]}
            `;

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
          const batch = vectors.slice(i, i + BATCH_SIZE);
          await index.upsert(batch);
          console.log(`Upserted batch ${i / BATCH_SIZE + 1}`);
        }

        console.log("✅ Hospital embeddings stored/updated in Pinecone.");
      });
  } catch (error) {
    console.error("❌ Error storing hospital embeddings:", error.message);
  }
}

/* ------------------ Pinecone Query ------------------ */

async function queryPinecone(query) {
  const queryEmbedding = await generateEmbeddings(query);

  const results = await index.query({
    vector: queryEmbedding,
    topK: 10,
    includeMetadata: true,
  });

  return results;
}

/* ------------------ Generate Response ------------------ */

async function generateResponse(query, history = []) {
  try {
    const results = await queryPinecone(query);

    if (!results || results.matches.length === 0) {
      return "Sorry, I could not find any relevant information.";
    }

    const context = results.matches
      .map((match) => {
        const m = match.metadata;
        return `Hospital: ${m.name}
          Address: ${m.address}
          City: ${m.city}`;
          })
      .join("\n\n");

    const historyText = history
      .map((h) => `${h.role === "user" ? "User" : "Assistant"}: ${h.text}`)
      .join("\n");

    const prompt = `You are an AI assistant for a hospital directory system.

You help users find hospitals based on name, city, or address.

Rules:
- Do NOT give medical advice.
- Only provide hospital information from the context.

Conversation so far:
${historyText}

Context:
${context}

User question:
${query}

If not found reply:
"Sorry, I could not find any matching hospital in the database."`;

    const result = await model.generateContent(prompt);
    const response = await result.response;

    return response.text();
  } catch (error) {
    console.error("❌ Error generating response:", error.message);
    return "Sorry, there was a problem generating the response.";
  }
}

/* ------------------ Routes ------------------ */

/* Initialize embeddings */

app.post("/initialize", async (req, res) => {
  const token = req.headers.authorization;

  if (token !== `Bearer ${process.env.ADMIN_SECRET}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  try {
    await loadEmbeddings();
    res.status(200).send("✅ Pinecone index initialized successfully.");
  } catch (error) {
    res.status(500).send("❌ Initialization failed: " + error.message);
  }
});

/* Intro route */

app.post("/start", async (req, res) => {
  try {

    const sessionId = uuidv4();

    sessions[sessionId] = {
      history: []
    };

    const audioBuffer = await textToSpeech(INTRO_MESSAGE);

    if (!audioBuffer || audioBuffer.length === 0) {
      throw new Error("TTS returned empty audio");
    }

    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("session-id", sessionId);

    res.end(audioBuffer);

  } catch (err) {

    console.error("START ROUTE ERROR:", err);

    res.status(500).send("Failed to generate intro speech");

  }
});
/* Query route */

app.post("/query", async (req, res) => {
  try {
    let { question, sessionId } = req.body;

    if (!question) {
      return res.status(400).send("❗ Please provide a question.");
    }

    // if (!sessionId) {
    //   sessionId = uuidv4();
    //   sessions[sessionId] = [];
    // }

    let isNewSession = false;

    if (!sessionId) {
      sessionId = uuidv4();
      sessions[sessionId] = [];
      isNewSession = true;
    }

    const history = sessions[sessionId] || [];

    // const response = await generateResponse(question, history);
    let response;

    if (isNewSession) {
      response = INTRO_MESSAGE;
    } else {
      response = await generateResponse(question, history);
    }

    history.push({ role: "user", text: question });
    history.push({ role: "assistant", text: response });

    sessions[sessionId] = history.slice(-MAX_HISTORY * 2);

    res.json({ response, sessionId });
  } catch (error) {
    res.status(500).send("❌ Query failed: " + error.message);
  }
});

/* Voice route */

app.post("/voice", upload.single("audio"), async (req, res) => {
  try {
    console.log("---- VOICE PIPELINE START ----");

    console.log("Step 1: Checking uploaded audio");

    if (!req.file) {
      console.log("❌ No audio file received");
      return res.status(400).send("No audio file uploaded");
    }

    console.log("✅ Audio received:", req.file.path);

    const fileBuffer = fs.readFileSync(req.file.path);
    console.log("Step 2: Audio buffer size:", fileBuffer.length);

    const sttForm = new FormData();
    sttForm.append("model_id", "scribe_v2");
    sttForm.append("file", new Blob([fileBuffer]), "audio.webm");

    console.log("Step 3: Sending audio to STT...");

    const sttRes = await fetch("https://api.elevenlabs.io/v1/speech-to-text", {
      method: "POST",
      headers: {
        "xi-api-key": process.env.ELEVENLABS_KEY,
      },
      body: sttForm,
    });

    const sttData = await sttRes.json();
    // console.log("STT raw response:", sttData);
    console.log("STT status:", sttRes.status);

    const userText = sttData.text;
    console.log("Step 4: Transcribed text:", userText);

    const sessionId = req.query.sessionId;

    console.log("Session received:", sessionId);
    console.log("Available sessions:", Object.keys(sessions));

  if (!sessionId || !sessions[sessionId]) {
    return res.status(400).send("Session not initialized. Please call /start first.");
  }

    // console.log("User said:", userText);
    
    // let answer;

    // if (isNewSession || !sessions[sessionId].introDone) {
    //   sessions[sessionId].introDone = true;
    //   answer = INTRO_MESSAGE;
    // } else if (!userText || userText.trim() === "") {
    //   answer = INTRO_MESSAGE;
    // } else {
    //   answer = await generateResponse(userText);
    // }

    console.log("Step 5: Generating AI response...");

    let answer = await generateResponse(userText);
    answer = answer.slice(0, 500);
    console.log("AI answer:", answer);

    console.log("Step 6: Sending text to TTS...");
    const ttsRes = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${process.env.ELEVENLABS_VOICE_ID}`,
      {
        method: "POST",
        headers: {
          "xi-api-key": process.env.ELEVENLABS_KEY,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: answer,
          model_id: "eleven_multilingual_v2",
        }),
      }
    );
    
    console.log("TTS status:", ttsRes.status);
    const audioArrayBuffer = await ttsRes.arrayBuffer();
    const audioBuffer = Buffer.from(audioArrayBuffer);

    console.log("Step 7: Generated audio size:", audioBuffer.length);

    fs.unlinkSync(req.file.path);

    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Content-Length", audioBuffer.length);

    console.log("Step 8: Sending audio response to frontend");
    console.log("---- VOICE PIPELINE END ----");
    
    res.setHeader("session-id", sessionId);
    res.end(audioBuffer);
  } catch (err) {
    console.error("VOICE ERROR:", err);
    res.status(500).send("Voice pipeline failed");
  }
});

/* Health check */

app.get("/health", async (req, res) => {
  try {
    await index.describeIndexStats();
    res.status(200).json({ status: "healthy" });
  } catch (error) {
    res.status(500).json({ status: "unhealthy", error: error.message });
  }
});

/* ------------------ Start Server ------------------ */

const PORT = process.env.PORT || 3000;

app.listen(PORT, "0.0.0.0", async () => {
  console.log(`🚀 Server running on port ${PORT}`);

  if (process.env.AUTO_INITIALIZE === "True") {
    try {
      await loadEmbeddings();
    } catch (err) {
      console.error("❌ Auto-initialization failed:", err);
    }
  }
});