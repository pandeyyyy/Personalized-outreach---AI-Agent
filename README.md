# Outreach Research Agent

A Streamlit app that researches a prospect, verifies claims against evidence, and creates an outreach draft for human review. It does not send emails.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Enter a Gemini API key in the app's sidebar to enable AI-assisted analysis and drafting. The key can also be provided through the `GEMINI_API_KEY` environment variable. The model can be changed with `GEMINI_MODEL`.

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository, select the branch and set the app file to `app.py`.
3. Add `GEMINI_API_KEY` under the app's **Settings > Secrets** if you want the key supplied through the environment. Alternatively, enter it in the app's sidebar.

The local virtual environments and generated `out/` files are excluded from Git.
