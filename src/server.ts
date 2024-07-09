import express from "express";
import { Request, Response } from "express";
import axios from "axios";
import multer from "multer";
import FormData from "form-data";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(express.json());

interface Message {
  role: string;
  content: string;
}

interface RequestPayload {
  model: string;
  messages: Message[];
}

interface TextToSpeechRequestPayload {
  model: string;
  input: string;
  voice: string;
}

interface ImageRequestPayload {
  model: string;
  prompt: string;
  size: string;
  quality: string;
  n: number;
}

interface EmbeddingRequestPayload {
  model: string;
  input: string[];
}

async function sendRequest(payload: RequestPayload): Promise<string> {
  const apiKey = payload.model.startsWith("claude")
    ? process.env.ANTHROPIC_API_KEY
    : process.env.OPENAI_API_KEY;

  const url = payload.model.startsWith("claude")
    ? "https://api.anthropic.com/v1/complete"
    : "https://api.openai.com/v1/chat/completions";

  try {
    const response = await axios.post(url, payload, {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
    });
    return JSON.stringify(response.data);
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
}

app.post("/generate-chat", async (req: Request, res: Response) => {
  try {
    const payload: RequestPayload = req.body;
    const responseJson = await sendRequest(payload);
    const response = JSON.parse(responseJson);

    const generatedChat = payload.model.startsWith("claude")
      ? response.completion
      : response.choices[0].message.content;

    res.status(200).send(generatedChat);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
});

app.post(
  "/transcribe-speech",
  upload.single("file"),
  async (req: Request, res: Response) => {
    try {
      const file = req.file;
      if (!file) {
        return res.status(400).send("No file uploaded");
      }

      const formData = new FormData();
      formData.append("model", "whisper-1");
      formData.append("file", file.buffer, { filename: "recording.wav" });

      const response = await axios.post(
        "https://api.openai.com/v1/audio/transcriptions",
        formData,
        {
          headers: {
            ...formData.getHeaders(),
            Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          },
        }
      );

      res.status(200).send(response.data);
    } catch (error) {
      console.error("Error:", error);
      res.status(500).send("Internal Server Error");
    }
  }
);

app.post("/generate-speech", async (req: Request, res: Response) => {
  try {
    const payload: TextToSpeechRequestPayload = req.body;
    const response = await axios.post(
      "https://api.openai.com/v1/audio/speech",
      payload,
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
        responseType: "arraybuffer",
      }
    );

    res.set("Content-Type", "audio/mpeg");
    res.status(200).send(response.data);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
});

app.post("/generate-image", async (req: Request, res: Response) => {
  try {
    const payload: ImageRequestPayload = req.body;
    const response = await axios.post(
      "https://api.openai.com/v1/images/generations",
      payload,
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
      }
    );

    const imageUrl = response.data.data[0].url;
    res.status(200).send(imageUrl);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
});

app.post("/get-embeddings", async (req: Request, res: Response) => {
  try {
    const payload: EmbeddingRequestPayload = req.body;
    const response = await axios.post(
      "https://api.openai.com/v1/embeddings",
      payload,
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
      }
    );

    res.status(200).json(response.data);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
});

app.post("/calculate-similarity", async (req: Request, res: Response) => {
  try {
    const { prompt, guess } = req.body;
    const payload: EmbeddingRequestPayload = {
      model: "text-embedding-ada-002",
      input: [prompt, guess],
    };

    const response = await axios.post(
      "https://api.openai.com/v1/embeddings",
      payload,
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
      }
    );

    const embeddings = response.data.data;
    const promptEmbedding = embeddings[0].embedding;
    const guessEmbedding = embeddings[1].embedding;

    const similarity = cosineSimilarity(promptEmbedding, guessEmbedding);
    const score = Math.round(similarity * 50 + 50);

    res.status(200).send(score.toString());
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
});

function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, _, i) => sum + a[i] * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
