import express from "express";
import cors from "cors";
import { HfInference } from "@huggingface/inference";
import multer from "multer";
import fs from "fs";
import dotenv from "dotenv";
import axios from "axios";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 9001;
const HF_TOKEN = process.env.HF_TOKEN;

if (!HF_TOKEN) {
    console.error("Missing Hugging Face API token! Set HF_TOKEN in .env");
    process.exit(1);
}

const inference = new HfInference(HF_TOKEN);
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json());

// Helper function to fetch website content
const fetchWebsiteContent = async (url) => {
    try {
        const response = await axios.get(url);
        return response.data;
    } catch (error) {
        console.error("Error fetching website content:", error);
        throw new Error("Failed to fetch website content");
    }
};

// Helper function to extract text from HTML (basic implementation)
const extractTextFromHTML = (html) => {
    const cleanedHTML = html.replace(/<script.*?>.*?<\/script>/gi, "").replace(/<style.*?>.*?<\/style>/gi, "");
    const text = cleanedHTML.replace(/<[^>]+>/g, "").trim();
    return text;
};
// Text Summarization and Translation Endpoint
app.post("/summarize-text", async (req, res) => {
  try {
      const { text, tgtLang } = req.body;

      if (!text || !tgtLang) {
          return res.status(400).json({ error: "Missing required fields: text or tgtLang" });
      }

      if (text.trim().length === 0) {
          throw new Error("No valid text content provided");
      }

      const summaryResponse = await inference.summarization({
          model: "google/pegasus-xsum",
          inputs: text,
          parameters: {
              max_length: 70,
              min_length: 50,
          },
      });

      if (!summaryResponse || !summaryResponse.summary_text) {
          throw new Error("Summarization failed: No summary text returned");
      }

      const summary = summaryResponse.summary_text;

      const translationResponse = await inference.translation({
          model: "facebook/nllb-200-distilled-600M",
          inputs: summary,
          parameters: { src_lang: "eng_Latn", tgt_lang: tgtLang },
      });

      res.json({ translatedSummary: translationResponse.translation_text });
  } catch (error) {
      console.error("Summarization Error:", error);
      res.status(500).json({ error: error.message || "Summarization and translation failed" });
  }
});


// Summarization Endpoint
app.post("/summarize", async (req, res) => {
    try {
        const { url, tgtLang } = req.body;

        if (!url || !tgtLang) {
            return res.status(400).json({ error: "Missing required fields: url or tgtLang" });
        }

        const htmlContent = await fetchWebsiteContent(url);
        const textContent = extractTextFromHTML(htmlContent);

        if (!textContent || textContent.trim().length === 0) {
            throw new Error("No text content found on the website");
        }

        const summaryResponse = await inference.summarization({
            model: "google/pegasus-xsum",
            inputs: textContent,
            parameters: {
                max_length: 70,
                min_length: 50,
            },
        });

        if (!summaryResponse || !summaryResponse.summary_text) {
            throw new Error("Summarization failed: No summary text returned");
        }

        const summary = summaryResponse.summary_text;

        const translationResponse = await inference.translation({
            model: "facebook/nllb-200-distilled-600M",
            inputs: summary,
            parameters: { src_lang: "eng_Latn", tgt_lang: tgtLang },
        });

        const translatedSummary = translationResponse.translation_text;

        res.json({ summary: translatedSummary });
    } catch (error) {
        console.error("Summarization Error:", error);
        res.status(500).json({ error: error.message || "Summarization and translation failed" });
    }
});

// Speech-to-Text Endpoint
app.post("/speech-to-text", upload.single("audio"), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No audio file uploaded" });
    }

    try {
        const audioData = fs.readFileSync(req.file.path);

        const response = await inference.automaticSpeechRecognition({
            model: "facebook/wav2vec2-large-960h",
            data: audioData,
        });

        fs.unlinkSync(req.file.path);

        if (!response || !response.text) {
            return res.status(500).json({ error: "Speech-to-text failed" });
        }

        res.json({ transcription: response.text });
    } catch (error) {
        console.error("Speech-to-Text Error:", error);

        if (req.file) {
            fs.unlink(req.file.path, () => {});
        }

        res.status(500).json({ error: "Speech-to-text conversion failed" });
    }
});

// Translation Endpoint
app.post("/translate", async (req, res) => {
    try {
        const { text, srcLang, tgtLang } = req.body;

        if (!text || !srcLang || !tgtLang) {
            return res.status(400).json({ error: "Missing required fields" });
        }

        const response = await inference.translation({
            model: "facebook/nllb-200-distilled-600M",
            inputs: text,
            parameters: { src_lang: srcLang, tgt_lang: tgtLang },
        });

        res.json({ translatedText: response.translation_text });
    } catch (error) {
        console.error("Translation Error:", error);
        res.status(500).json({ error: "Translation failed" });
    }
});

// Simple Question Answering (using Hugging Face model directly)
app.post("/qa", async (req, res) => {
    try {
        const { url, question } = req.body;

        if (!url || !question) {
            return res.status(400).json({ error: "Missing required fields: url or question" });
        }

        const htmlContent = await fetchWebsiteContent(url);
        const textContent = extractTextFromHTML(htmlContent);

        if (!textContent || textContent.trim().length === 0) {
            throw new Error("No text content found on the website");
        }

        const qaResponse = await inference.questionAnswering({
            model: "deepset/roberta-base-squad2", // A common QA model
            inputs: {
                question: question,
                context: textContent,
            },
        });

        if (!qaResponse || !qaResponse.answer) {
            return res.status(500).json({ error: "Question answering failed" });
        }

        res.json({ answer: qaResponse.answer });
    } catch (error) {
        console.error("Question Answering Error:", error);
        res.status(500).json({ error: error.message || "Question answering failed" });
    }
});

// Start Server
app.listen(PORT, () => {
    console.log(`✅ Server running on port ${PORT}`);
});