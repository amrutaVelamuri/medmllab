export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();
  const { prompt } = req.body;
  if (!prompt) return res.status(400).json({ error: "No prompt" });

  const MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3-12b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
  ];

  const errors = [];
  for (const model of MODELS) {
    try {
      const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          max_tokens: 1000,
        }),
      });
      const data = await response.json();
      if (data.error) { errors.push(`${model}: ${JSON.stringify(data.error)}`); continue; }
      const text = data.choices?.[0]?.message?.content || "";
      if (text) return res.status(200).json({ text });
    } catch (e) { errors.push(`${model}: ${e.message}`); continue; }
  }

  res.status(500).json({ error: errors.join(" | ") });
}
