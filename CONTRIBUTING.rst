============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at the project's `issues page`_.

When reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

.. _issues page: https://github.com/NannyML/nannyml/issues

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it. Issues labeled "good first issue"
are well suited for first time contributors.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

NannyML could always use more documentation, whether as part of the
official NannyML docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

Feel free to give feedback on anything related to NannyML. There are two ways to give feedback.
The first is to chat with us in our `slack community`_.

The second way is to file an issue at the project's `issues page`_.

If you are proposing a feature:

- Describe the problem the feature will solve.
- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.


  .. _slack community: https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ

Get Started!
------------

Ready to contribute code? Here's how to set up `nannyml` for local development.

1. Fork the `nannyml` repo on GitHub.
2. Clone your fork locally ::

    $ git clone git@github.com:your_name_here/nannyml.git

3. Ensure poetry_ is installed.

.. note::

    When installing poetry on Mac OSX Monterey, if you get a permission denied error for ``.zshrc``,
    you will have to add the following manually using ``sudo``: ``export PATH="$HOME/.poetry/bin:$PATH”``

4. Install dependencies and start your virtualenv. Execute the following from within your local repository directory: ::

    $ poetry install -E test -E doc -E dev

5. Create a branch for local development: ::

    $ git checkout -b name-of-your-bugfix-or-feature

  Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox. (“pip install tox” if it is not already installed): ::

    $ poetry run tox

7. Commit your changes and push your branch to GitHub: ::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.


.. _poetry: https://python-poetry.org/docs/

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in ``README.md``.
3. The pull request should work for Python 3.7, 3.8, 3.9 and 3.10. Check
   the project's `github actions page`_ and make sure that the tests pass
   for all supported Python versions.

.. _`github actions page`: https://github.com/NannyML/nannyml/actions

Tips
----

::

$ poetry run pytest tests/test_nannyml.py

To run a subset of tests.


Deploying
----------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in ``CHANGELOG.md``).
Then run: ::

$ poetry run bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags


GitHub Actions will then deploy to PyPI if tests pass.
