"""Apple's LLM exporter with this project's ports registered.

Usage: ./run.sh python -m coreai_ports.export <hf_model_id> --platform iOS [Apple export options]
"""

from coreai_models.llm.export import main

from coreai_ports.register import register

if __name__ == "__main__":
    register()
    main()
