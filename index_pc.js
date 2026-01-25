import express from "express";
import cors from "cors";
import fs from "fs";
// Store new vectors
import csv from "csv-parser";

import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";
import { v4 as uuidv4 } from "uuid";

const sessions = {};
const MAX_HISTORY = 5; // last 5 turns

dotenv.config();

const app = express();
app.use(express.json());

// app.use(
//   cors({
//     origin: "http://127.0.0.1:5500/fe.html",
//     methods: ["GET", "POST", "OPTIONS"],
//     credentials: true,
//   }),
// );
app.use(cors());

async function textToSpeech(text) {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${process.env.ELEVENLABS_VOICE_ID}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": process.env.ELEVENLABS_API_KEY,
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
    },
  );

  const buffer = await response.arrayBuffer();
  return Buffer.from(buffer);
}

// ------------------ Setup Embeddings and Gemini ------------------

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

// ------------------ Setup Pinecone ------------------

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

// Delete previous vectors
async function deleteOldVectors() {
  try {
    const stats = await index.describeIndexStats();
    const ids = Object.keys(stats.namespaces?.[""]?.vectorCount ?? {}).map(
      (id) => id.toString(),
    );
    if (ids.length > 0) {
      await index.delete1({ ids });
      console.log(`🗑️ Deleted ${ids.length} old vectors.`);
    }
  } catch (error) {
    console.error("Error deleting old vectors:", error.message);
  }
}

async function loadEmbeddings() {
  try {
    const hospitals = [];

    fs.createReadStream("hospital_data.csv")
      .pipe(csv())
      .on("data", (row) => {
        hospitals.push(row);
      })
      .on("end", async () => {
        await deleteOldVectors();

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

        console.log("✅ Hospital embeddings stored in Pinecone.");
      });
  } catch (error) {
    console.error("❌ Error storing hospital embeddings:", error.message);
  }
}

// ------------------ Query Pinecone ------------------

async function queryPinecone(query) {
  const queryEmbedding = await generateEmbeddings(query);

  const results = await index.query({
    vector: queryEmbedding,
    topK: 10,
    includeMetadata: true,
  });

  return results;
}

async function generateResponse(query, history=[]) {
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

If not found, reply:
"Sorry, I could not find any matching hospital in the database."`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text();
    // return prompt;
  } catch (error) {
    console.error("❌ Error generating response:", error.message);
    return "Sorry, there was a problem generating the response.";
  }
}

// ------------------ Routes ------------------

app.post("/initialize", async (req, res) => {
  // Optional: protect this with a secret
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

app.post("/query", async (req, res) => {
  try {
    let { question, sessionId } = req.body;

    if (!question) {
      return res.status(400).send("❗ Please provide a question.");
    }

    // create new session if not provided
    if (!sessionId) {
      sessionId = uuidv4();
      sessions[sessionId] = [];
    }

    const history = sessions[sessionId] || [];

    const response = await generateResponse(question, history);

    // update history
    history.push({ role: "user", text: question });
    history.push({ role: "assistant", text: response });

    // keep only last N
    sessions[sessionId] = history.slice(-MAX_HISTORY * 2);

    res.json({ response, sessionId });
  } catch (error) {
    res.status(500).send("❌ Query failed: " + error.message);
  }
});

import multer from "multer";
const upload = multer({ dest: "uploads/" });

// app.post("/voice", upload.single("audio"), async (req, res) => {
//   try {
//     const fileBuffer = fs.readFileSync(req.file.path);

//     const form = new FormData();
//     form.append("model_id", "scribe_v2");
//     form.append("file", new Blob([fileBuffer]), "audio.webm");

//     const response = await fetch("https://api.elevenlabs.io/v1/speech-to-text", {
//       method: "POST",
//       headers: {
//         "xi-api-key": process.env.ELEVENLABS_KEY
//       },
//       body: form
//     });

//     const data = await response.json();
//     console.log("FULL RESPONSE:", data);
//     console.log("Transcript:", data.text);
//     const answer = await generateResponse(userText);

//     fs.unlinkSync(req.file.path);
//     res.json({ transcript: data.text });
//   } catch (err) {
//     console.error(err);
//     res.status(500).send("Error");
//   }
// });
app.post("/voice", upload.single("audio"), async (req, res) => {
  try {
    console.log("---- VOICE PIPELINE START ----");

    // ========== STT ==========
    const fileBuffer = fs.readFileSync(req.file.path);
    console.log("Audio file size:", fileBuffer.length);

    const sttForm = new FormData();
    sttForm.append("model_id", "scribe_v2");
    sttForm.append("file", new Blob([fileBuffer]), "audio.webm");

    const sttRes = await fetch("https://api.elevenlabs.io/v1/speech-to-text", {
      method: "POST",
      headers: {
        "xi-api-key": process.env.ELEVENLABS_KEY,
      },
      body: sttForm,
    });

    console.log("STT status:", sttRes.status);
    const sttData = await sttRes.json();
    console.log("STT raw:", sttData);

    const userText = sttData.text;
    console.log("User said:", userText);

    // ========== RAG ==========
    let answer = await generateResponse(userText);
    // const answer = "this is a test answer"
    answer = answer.slice(0, 100);
    console.log("AI answer:", answer);

    // ========== TTS ==========
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
      },
    );

    console.log("TTS status:", ttsRes.status);

    const audioArrayBuffer = await ttsRes.arrayBuffer();
    const audioBuffer = Buffer.from(audioArrayBuffer);
    console.log("TTS audio size:", audioBuffer.length);

    fs.unlinkSync(req.file.path);

    // ========== SEND AUDIO ==========
    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Content-Length", audioBuffer.length);
    res.setHeader("Cache-Control", "no-cache");

    console.log("---- VOICE PIPELINE END ----");
    res.end(audioBuffer);
  } catch (err) {
    console.error("VOICE ERROR:", err);
    res.status(500).send("Voice pipeline failed");
  }
});

app.get("/health", async (req, res) => {
  try {
    await index.describeIndexStats();
    res.status(200).json({ status: "healthy" });
  } catch (error) {
    res.status(500).json({ status: "unhealthy", error: error.message });
  }
});

// ------------------ Start Server ------------------

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
