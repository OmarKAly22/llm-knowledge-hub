import os
from flask import Flask, render_template, request, jsonify
import markdown2

app = Flask(__name__)

# Configuration
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "dev-secret-key-change-in-production"
)

# Set debug mode based on environment
if os.environ.get("FLASK_ENV") == "production":
    app.config["DEBUG"] = False
else:
    app.config["DEBUG"] = True


# Helper function to load markdown files
def load_markdown(filename):
    """Load and convert markdown file to HTML."""
    filepath = os.path.join("docs", "guides", filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            return markdown2.markdown(
                content, extras=["fenced-code-blocks", "tables", "header-ids"]
            )
    return None


# Routes
@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")


@app.route("/llm-guide")
def llm_guide():
    """LLM Guide page."""
    content = load_markdown("llm-guide.md")
    return render_template("llm_guide.html", content=content)


@app.route("/llm-qa")
def llm_qa():
    """LLM Q&A page."""
    content = load_markdown("llm-qa.md")
    return render_template("llm_qa.html", content=content)


@app.route("/agentic-guide")
def agentic_guide():
    """Agentic AI Guide page."""
    content = load_markdown("agentic-ai-guide.md")
    return render_template("agentic_guide.html", content=content)


@app.route("/agentic-qa")
def agentic_qa():
    """Agentic AI Q&A page."""
    content = load_markdown("agentic-ai-qa.md")
    return render_template("agentic_qa.html", content=content)


@app.route("/api/search", methods=["POST"])
def search():
    """
    API endpoint for search functionality.

    Expected JSON body:
    {
        "query": "search term"
    }
    """
    data = request.get_json()
    query = data.get("query", "").lower()

    # TODO: Implement actual search logic
    # For now, return empty results
    results = []

    return jsonify({"query": query, "results": results, "count": len(results)})


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return "Internal Server Error", 500


# Health check endpoint for monitoring
@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    # Get port from environment variable (for deployment platforms)
    port = int(os.environ.get("PORT", 5000))

    # Run the app
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
