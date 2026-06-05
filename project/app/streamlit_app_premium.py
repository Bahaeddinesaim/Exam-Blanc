"""Compatibility entrypoint for Streamlit Cloud.

The canonical dashboard lives in streamlit_app.py. Some deployments may still
point to this file, so this wrapper launches the maintained app.
"""

from streamlit_app import main


main()
