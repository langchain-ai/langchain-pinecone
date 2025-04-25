# Contributing to LangChain-Pinecone

Please feel free to contribute new features, bug fixes, or documentation. We're always eager to hear your suggestions.

Please follow these guidelines when making a contribution:
1. **Check for Existing Issues:** Before making any changes, [check here for related issues](https://github.com/langchain-ai/langchain-pinecone/issues).
2. **Run Your Changes by Us!** If no related issue exists yet, please create one and suggest your changes. Checking in with the team first will allow us to determine if the changes are in scope.
3. **Set Up Development Environment** If the changes are agreed, then you can go ahead and set up a development environment (see [Setting Up Your Development Environment](#setting-up-your-development-environment) below).
4. **Create an Early Draft Pull Request** Once you have commits ready to be shared, initiate a draft Pull Request with an initial version of your implementation and request feedback. It's advisable not to wait until the feature is fully completed.
5. **Ensure that All Pull Request Checks Pass** There are Pull Request checks that need to be satisfied before the changes can be merged. These appear towards the bottom of the Pull Request webpage on GitHub, and include:
    - Ensure that the Pull Request title is prepended with a [valid type](https://flank.github.io/flank/pr_titles/). E.g. `feat: My New Feature`.
    - Run linting (and fix any issues that are flagged) by:
        - Navigating to `/libs/pinecone`
        - Running `make lint` to check for linting issues
        - Running `make format` to automatically fix formatting issues
        - Confirming the linters pass using `make lint` again
    - Ensure that, for any new code, new PyTests are written in the appropriate test directory (`/libs/pinecone/tests/unit_tests` or `/libs/pinecone/tests/integration_tests`). If any code is removed, ensure that corresponding PyTests are also removed. Verify all tests pass by running:
        - `make test` for unit tests
        - `make integration_tests` for integration tests (when applicable)

> **Feedback and Discussion:**
While we encourage you to initiate a draft Pull Request early to get feedback on your implementation, we also highly value discussions and questions. If you're unsure about any aspect of your contribution or need clarification on the project's direction, please don't hesitate to use the [Issues section](https://github.com/langchain-ai/langchain-pinecone/issues) of our repository. Engaging in discussions or asking questions before starting your work can help ensure that your efforts align well with the project's goals and existing work.

# Setting Up Your Development Environment

1. Fork on GitHub:
    Go to the [repository's page](https://github.com/langchain-ai/langchain-pinecone) on GitHub: 
    Click the "Fork" button in the top-right corner of the page.

2. Clone Your Fork:
    After forking, you'll be taken to your new fork of the repository on GitHub. Copy the URL of your fork from the address bar or by clicking the "Code" button and copying the URL under "Clone with HTTPS" or "Clone with SSH".
    Open your terminal or command prompt.
    Use the git clone command followed by the URL you copied to clone the repository to your local machine. Replace `https://github.com/<your-gh-username>/langchain-pinecone.git` with the URL of your fork:
    ```
    git clone https://github.com/<your-gh-username>/langchain-pinecone.git
    ```

3. Ensure you have [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/), for macOS use `brew install uv` or for Linux (and Mac) use `curl -LsSf https://astral.sh/uv/install.sh | sh`.

4. Then navigate to the cloned folder and the specific library folder:
    ```
    # Move into the cloned folder
    cd langchain-pinecone/

    # Move into the library directory
    cd libs/pinecone/

    # Create a virtual environment (if using uv)
    uv venv --python 3.12.7  # python arg is optional
    # Activate venv
    source .venv/bin/activate
    # install all dev dependencies
    uv sync --group test --group codespell --group test_integration --group lint --group dev --group typing
    ```