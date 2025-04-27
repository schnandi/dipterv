from app import create_app

app = create_app()

if __name__ == "__main__":
    # Run the Flask development server with debugging enabled.
    app.run(debug=True)
