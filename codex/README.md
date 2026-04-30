# Metaphor Creator

A small Streamlit app that generates, evaluates, and improves metaphors for a complex concept and a target audience.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add your API key to `.env` in this folder:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional:

```powershell
$env:OPENAI_MODEL="gpt-5.4-mini"
```

## Run

```powershell
streamlit run app.py
```

The app will open in your browser. Enter a concept and audience, optionally choose a tone and category, and click **Generate metaphors**. Each run shows token usage and an estimated cost when pricing is available for the selected model.
