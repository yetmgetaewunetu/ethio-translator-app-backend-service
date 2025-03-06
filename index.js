import express from "express";
import cors from "cors";
import { HfInference } from "@huggingface/inference";
import multer from "multer";
import fs from "fs";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 9001;
const HF_TOKEN = process.env.HF_TOKEN; // Hugging Face API token

if (!HF_TOKEN) {
  console.error("Missing Hugging Face API token! Set HF_TOKEN in .env");
  process.exit(1);
}

const inference = new HfInference(HF_TOKEN);
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json());

// Speech-to-Text Endpoint
app.post("/speech-to-text", upload.single("audio"), async (req, res) => {
    
  if (!req.file) {
    return res.status(400).json({ error: "No audio file uploaded" });
  }

  try {
    const audioData = fs.readFileSync(req.file.path);


    const response = await inference.automaticSpeechRecognition({
      model: "facebook/wav2vec2-large-960h", // Use a supported model
      data: audioData,
    });

    

    // Clean up the uploaded file
    fs.unlinkSync(req.file.path);

    if (!response || !response.text) {
      return res.status(500).json({ error: "Speech-to-text failed" });
    }

    console.log(response.text)
    res.json({ transcription: response.text });
  } catch (error) {
    console.error("Speech-to-Text Error:", error);

    // Ensure the uploaded file is deleted even in case of an error
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

// Start Server
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});