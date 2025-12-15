Contributing
============

We welcome contributions to the Decline Curve Analysis library! This guide will help you get started.

Development Setup
-----------------

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/yourusername/decline-analysis.git
      cd decline-analysis

2. **Create Development Environment**

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

3. **Install Development Dependencies**

   .. code-block:: bash

      pip install pytest pytest-cov black isort flake8 mypy

Running Tests
-------------

Run the full test suite:

.. code-block:: bash

   pytest tests/ -v

Run with coverage:

.. code-block:: bash

   pytest tests/ --cov=decline_curve --cov-report=html

Code Style
----------

We use several tools to maintain code quality:

**Black** for code formatting:

.. code-block:: bash

   black decline_curve/ tests/

**isort** for import sorting:

.. code-block:: bash

   isort decline_curve/ tests/

**flake8** for linting:

.. code-block:: bash

   flake8 decline_curve/ tests/

**mypy** for type checking:

.. code-block:: bash

   mypy decline_curve/

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs/
   make html

View documentation:

.. code-block:: bash

   open _build/html/index.html

Contribution Guidelines
-----------------------

1. **Fork and Branch**: Create a feature branch from main
2. **Write Tests**: All new features should include tests
3. **Update Documentation**: Add docstrings and update docs as needed
4. **Follow Style**: Use black, isort, and follow existing patterns
5. **Write Clear Commits**: Use descriptive commit messages

Pull Request Process
--------------------

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG.md
4. Submit pull request with clear description
5. Address review feedback

Areas for Contribution
----------------------

- New forecasting models
- Additional evaluation metrics
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

Contact
-------

For questions about contributing, please open an issue on GitHub.
