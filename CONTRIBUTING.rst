============
Contributing
============

If you're reading this, you might be interested in contributing to NannyML.
Thanks for that, we're stoked to have you aboard!

If you're not a software engineer, don't fret. There are plenty other ways you can help us out,
which we equally value and appreciate!

Spread the word
----------------

Help us raising awareness to the importance of post-deployment model performance!

- **Share our project with your network**: you can post about us on your Social Media accounts,
  or talk about our project with your friends and co-workers. The more people use it, the more feedback we get
  and the faster we can improve our product!
- **Participate in the conversation**: express your opinion, concerns and feedback with the community on
  our Slack channel. Feel free to engage in the discussion of various topics we have there
  or simply participate in our polls.
- **Write about us**: by featuring us in your blog, newsletter or similar.
  The content you write about can include examples of where to use our library, how to use it (tutorial),
  or why you find it useful.
- **Engage with us**: we are active in multiple Social Media channels. A simple like or comment can help us
  understand what you like seeing, while reaching a bigger audience.
- **Contribute to our internationalisation**: translate our documentation to your native language to help people
  understand better.

Be a part of the team
---------------------

- **Organise our issues**: by linking duplicates and suggesting new labels.
- **Keep our issues updated:** by suggesting closing old issues, or asking clarifying questions to open ones.
- **Answer questions about the project:** while going through our open issues, and moderate discussions in our Slack channel.
- **Organise events for us:** get in touch if you would like to organise a meetup,
  a conference or a get-together event on our behalf.
  You can also help us finding suitable events to submit proposals for speaking.
- **Suggest new layouts:** you can suggest improvements in our interface, or help us design
  cool swag to give away to the community.
- **Share new knowledge:** send us novel research or interesting content you might find relevant.
- **Review code:** submitted by other people, or offer to mentor another contributor.

Contribute to the codebase
--------------------------



- **Report bugs**: inform us of things breaking or improvements you'd like to see by submitting an issues on
  our `issues page`_. Just use one of the easy templates provided so we know all we have to know.
- **Smash some bugs**: look through the GitHub issues for bugs. Anything tagged with `bug` and
  `help wanted` is open to whoever wants to implement it. Issues labeled `good first issue`
  are well suited for first time contributors.
- **Implement features**: look through the GitHub issues for features. Anything tagged with "enhancement"
  and "help wanted" is open to whoever wants to implement it.
- **Write documentation**: it might not seem very glorious, but it is some of the hardest yet most important work to be
  done. Additions to the official NannyML docs, docstrings or even on the web in blog posts or articles are highly
  appreciated!
- **Submit feedback**: feel free to give feedback on anything related to NannyML, we love hearing what you like and
  what you don't like even more! There are two ways to give feedback:
  - Chat with us in our `slack community`_.
  - Create a feature request at our project's `issues page`_.



.. _issues page: https://github.com/NannyML/nannyml/issues
.. _slack community: https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ

Get started coding
~~~~~~~~~~~~~~~~~~

Ready to contribute code? Here's how to set up `nannyml` for local development.

1. Fork the `nannyml` repo on GitHub.
2. Clone your fork locally: ::

    $ git clone git@github.com:your_name_here/nannyml.git

3. Ensure poetry_ is installed.

.. note::

    When installing poetry on Mac OSX Monterey, if you get a permission denied error for ``.zshrc``,
    you will have to add the following manually using ``sudo``: ``export PATH="$HOME/.poetry/bin:$PATH”``

4. Install dependencies and start your virtualenv. Execute the following from within your local repository directory: ::

    $ poetry install

5. Create a branch for local development: ::

    $ git checkout -b name-of-your-bugfix-or-feature

  Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox: ::

    $ poetry run tox

7. Commit your changes and push your branch to GitHub: ::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.


.. _poetry: https://python-poetry.org/docs/

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~

::

$ poetry run pytest tests/test_nannyml.py

To run a subset of tests.
