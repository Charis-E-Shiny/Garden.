let session = null;
let stoi = {};
let itos = {};
const SPECIALS = { PAD: "<pad>", UNK: "<unk>", USER: "<user>", BOT: "<bot>", EOS: "<eos>" };

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");
const statusDiv = document.getElementById("status");

// Load ONNX Model and Vocab
async function init() {
  try {
    const vocabResp = await fetch("vocab.json");
    stoi = await vocabResp.json();
    for (const [k, v] of Object.entries(stoi)) {
      itos[v] = k;
    }

    session = await ort.InferenceSession.create("model.onnx", {
      executionProviders: ["wasm"]
    });

    statusDiv.textContent = "Model loaded! Ready to chat.";
    userInput.disabled = false;
    sendBtn.disabled = false;
  } catch (e) {
    statusDiv.textContent = "Error loading model or vocab. Check browser console.";
    console.error(e);
  }
}

// Tokenize & Detokenize matching tiny_gpt.py
function tokenize(text) {
  const matches = text.toLowerCase().match(/\w+(?:'\w+)*|[^\w\s]/g);
  return matches || [];
}

function detokenize(tokens) {
  let text = tokens.join(" ");
  text = text.replace(/\s+([.,!?;:%)])/g, "$1");
  text = text.replace(/([¿¡(])\s+/g, "$1");
  text = text.replace(/\bi\b/g, "I");
  text = text.replace(/(^|[.!?]\s+)([a-z])/g, (m, p1, p2) => p1 + p2.toUpperCase());
  return text;
}

// Softmax with Temperature and Top-K sampling
function sampleNextToken(logitsArray, vocabSize, bannedIds, temperature = 0.7, topK = 8) {
  let logits = Array.from(logitsArray);

  // Mask banned tokens
  bannedIds.forEach(id => { logits[id] = -Infinity; });

  if (temperature <= 0) {
    let maxIdx = 0, maxVal = -Infinity;
    logits.forEach((val, idx) => { if (val > maxVal) { maxVal = val; maxIdx = idx; } });
    return maxIdx;
  }

  // Apply temperature
  logits = logits.map(v => v / temperature);

  // Apply Top-K
  let indexed = logits.map((v, i) => ({ val: v, idx: i }));
  indexed.sort((a, b) => b.val - a.val);
  indexed = indexed.slice(0, topK);

  // Exponentiate & normalize
  const maxVal = indexed[0].val;
  const exps = indexed.map(item => Math.exp(item.val - maxVal));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExps);

  // Multinomial sample
  const rand = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (rand <= cum) return indexed[i].idx;
  }
  return indexed[0].idx;
}

// Generate response using ONNX Session
async function generateResponse(text) {
  const words = tokenize(text).slice(-40);
  const unkId = stoi[SPECIALS.UNK];
  const tokIds = words.map(w => (stoi[w] !== undefined ? stoi[w] : unkId));

  let prompt = [stoi[SPECIALS.USER], ...tokIds, stoi[SPECIALS.BOT]];
  const eosId = stoi[SPECIALS.EOS];
  const banned = [stoi[SPECIALS.PAD], unkId, stoi[SPECIALS.USER], stoi[SPECIALS.BOT]];

  const outTokens = [];
  const maxNewTokens = 40;
  const blockSize = 64;

  for (let i = 0; i < maxNewTokens; i++) {
    const window = prompt.slice(-blockSize);
    const inputTensor = new ort.Tensor("int64", BigInt64Array.from(window.map(BigInt)), [1, window.length]);
    
    const outputs = await session.run({ input_ids: inputTensor });
    const logitsData = outputs.logits.data; // Shape: [1, seq_len, vocab_size]
    const vocabSize = Object.keys(stoi).length;

    // Get logits for the last token position
    const lastTokenLogits = logitsData.slice((window.length - 1) * vocabSize, window.length * vocabSize);
    const nextToken = sampleNextToken(lastTokenLogits, vocabSize, banned);

    if (nextToken === eosId) break;
    prompt.push(nextToken);
    outTokens.push(itos[nextToken] || "");
  }

  return detokenize(outTokens) || "...";
}

// Append messages to UI
function appendMessage(sender, text) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", sender);
  msgDiv.textContent = text;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Send Message Handler
async function handleSend() {
  const text = userInput.value.trim();
  if (!text) return;

  appendMessage("user", text);
  userInput.value = "";
  userInput.disabled = true;

  const botReply = await generateResponse(text);
  appendMessage("bot", botReply);
  userInput.disabled = false;
  userInput.focus();
}

sendBtn.addEventListener("click", handleSend);
userInput.addEventListener("keypress", (e) => { if (e.key === "Enter") handleSend(); });

// Web Speech API for Voice Input
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => micBtn.classList.add("recording");
  recognition.onend = () => micBtn.classList.remove("recording");
  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    userInput.value = transcript;
    handleSend();
  };

  micBtn.addEventListener("click", () => recognition.start());
} else {
  micBtn.disabled = true;
  micBtn.title = "Speech Recognition is not supported in this browser.";
}

init();