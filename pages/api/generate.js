export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();
  const { prompt } = req.body;
  if (!prompt) return res.status(400).json({ error: "No prompt" });

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
      },
      body: JSON.stringify({
       model: "google/gemini-2.0-flash-exp:free",

        messages: [{ role: "user", content: prompt }],
        max_tokens: 1000,
      }),
    });
    const data = await response.json();
    if (data.error) return res.status(500).json({ error: data.error.message });
    const text = data.choices?.[0]?.message?.content || "";
    res.status(200).json({ text });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}
