import os
from flask import Flask, render_template, request, jsonify
import markdown2
import base64
import zlib
from pathlib import Path

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


def plantuml_encode(plantuml_text):
    """Encode PlantUML text for URL"""
    zlibbed_str = zlib.compress(plantuml_text.encode('utf-8'))
    compressed_string = zlibbed_str[2:-4]
    return base64.urlsafe_b64encode(compressed_string).decode('utf-8')

@app.route('/diagrams')
def diagrams_index():
    """Display all available diagrams"""
    # FIXED: Changed from 'docs/diagrams' to 'docs/diagrams/uml'
    diagrams_dir = Path('docs/diagrams/uml')
    
    # Organize diagrams by category
    categories = {
        'Agent Anatomy': [],
        'Agent Architectures': [],
        'Multi-Agent Systems': [],
        'Implementation Patterns': [],
        'Common Patterns': [],
        'Advanced Patterns': []
    }
    
    # Map filenames to categories
    category_map = {
        'agent-anatomy.puml': 'Agent Anatomy',
        'react-architecture.puml': 'Agent Architectures',
        'function-calling-architecture.puml': 'Agent Architectures',
        'planning-architecture.puml': 'Agent Architectures',
        'hierarchical-architecture.puml': 'Agent Architectures',
        'reflexion-architecture.puml': 'Agent Architectures',
        'multiagent-sequential.puml': 'Multi-Agent Systems',
        'multiagent-parallel.puml': 'Multi-Agent Systems',
        'multiagent-debate.puml': 'Multi-Agent Systems',
        'pattern-agent-loop.puml': 'Implementation Patterns',
        'pattern-tool-definition.puml': 'Implementation Patterns',
        'pattern-memory-management.puml': 'Implementation Patterns',
        'pattern-safety-control.puml': 'Implementation Patterns',
        'agent-research.puml': 'Common Patterns',
        'agent-code.puml': 'Common Patterns',
        'agent-data-analysis.puml': 'Common Patterns',
        'agent-customer-service.puml': 'Common Patterns',
        'agent-personal-assistant.puml': 'Common Patterns',
        'advanced-hierarchical.puml': 'Advanced Patterns',
        'advanced-memory-augmented.puml': 'Advanced Patterns',
        'advanced-learning.puml': 'Advanced Patterns',
    }
    
    # Read all .puml files
    if diagrams_dir.exists():
        for puml_file in diagrams_dir.glob('*.puml'):
            filename = puml_file.name
            if filename in category_map:
                category = category_map[filename]
                title = filename.replace('.puml', '').replace('-', ' ').title()
                categories[category].append({
                    'filename': filename,
                    'title': title,
                    'url': f'/diagrams/{filename.replace(".puml", "")}'
                })
    
    return render_template('diagrams_index.html', categories=categories)

@app.route('/diagrams/<diagram_name>')
def show_diagram(diagram_name):
    """Display a specific diagram"""
    # FIXED: Changed from 'docs/diagrams/{diagram_name}.puml' to 'docs/diagrams/uml/{diagram_name}.puml'
    puml_file = Path(f'docs/diagrams/uml/{diagram_name}.puml')
    
    if not puml_file.exists():
        return "Diagram not found", 404
    
    # Read the PlantUML file
    with open(puml_file, 'r', encoding='utf-8') as f:
        puml_content = f.read()
    
    # Encode for PlantUML server
    encoded = plantuml_encode(puml_content)
    
    # Generate URLs for different formats
    plantuml_server = "https://www.plantuml.com/plantuml"
    diagram_urls = {
        'svg': f"{plantuml_server}/svg/{encoded}",
        'png': f"{plantuml_server}/png/{encoded}",
        'txt': f"{plantuml_server}/txt/{encoded}"
    }
    
    title = diagram_name.replace('-', ' ').title()
    
    return render_template('diagram_view.html', 
                         diagram_name=diagram_name,
                         title=title,
                         diagram_urls=diagram_urls,
                         puml_content=puml_content)

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