"""Backend package marker.

Import `backend.app.create_app` explicitly; this module must stay free of
side effects so the ML subpackage can be tested without Flask installed.
"""
