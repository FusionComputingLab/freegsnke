Logging
=======

FreeGSNKE uses Python's standard :mod:`logging` module under a single root
package logger named ``"freegsnke"``. All library output—including machine
construction, static and evolutive solver progress, virtual circuits, and
formatted summaries—flows through module-scoped child loggers
(``freegsnke.<module>``).

Default behavior
----------------

Importing :mod:`freegsnke` attaches a default console handler that writes to
the current ``sys.stdout`` at ``INFO`` level using the format::

    %(levelname)s [%(name)s]: %(message)s

While the package-managed handler is active, ``propagate`` is set to ``False``
on the ``"freegsnke"`` logger so messages are not duplicated if a host
application also configures Python's root logger.

Configuring logging (users)
---------------------------

FreeGSNKE exports convenience helpers directly from the top-level package:

.. code-block:: python

    import freegsnke

    # Change verbosity globally across FreeGSNKE (accepts str or int levels)
    freegsnke.set_log_level("DEBUG")
    freegsnke.set_log_level("WARNING")

    # Customize stream and format (e.g. plain message-only output)
    freegsnke.setup_logging(level="INFO", fmt=freegsnke.SIMPLE_FORMAT)

    # Filter a specific submodule independently
    freegsnke.get_logger("GSstaticsolver").setLevel("WARNING")

Pure-library mode & ``pytest`` ``caplog``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you embed FreeGSNKE inside an application that configures Python's root
logger (for example via ``logging.basicConfig`` or a custom logging config), or
if you want to capture FreeGSNKE records with ``pytest``'s ``caplog`` fixture,
disable the default handler:

.. code-block:: python

    import logging
    import freegsnke

    freegsnke.disable_default_handler()
    logging.basicConfig(level=logging.INFO)

``disable_default_handler()`` removes the package-managed stdout handler,
attaches a :class:`logging.NullHandler`, and restores ``propagate = True`` so
records flow to the root logger. Call ``freegsnke.enable_default_handler()`` to
restore the default console handler.

Interaction with ``verbose`` and ``suppress``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solver and controller methods that accept ``verbose`` or ``suppress`` flags map
them directly onto log levels:

- ``verbose=True`` emits diagnostics at ``INFO``; ``verbose=False`` emits them
  at ``DEBUG``.
- ``suppress=False`` emits convergence summaries at ``INFO`` (or ``WARNING`` on
  non-convergence); ``suppress=True`` demotes them to ``DEBUG``.

At the default ``INFO`` level, ``verbose=False`` and ``suppress=True`` stay
quiet; lowering the package level to ``DEBUG`` unlocks both globally without
changing call sites.

Logging conventions (contributors)
----------------------------------

Direct stdout writes (``print``, ``pprint``, ``sys.stdout.write``,
``sys.stderr.write``) and root-logger calls (``logging.basicConfig``,
``logging.info``, etc.) are disallowed in all library modules under
``freegsnke/`` (excluding ``freegsnke/tests/``) and enforced by an AST-based
unit test in ``freegsnke/tests/test_logging.py``.

In library modules, obtain a module-scoped logger after all imports and use
lazy ``%``-style formatting:

.. code-block:: python

    import logging

    logger = logging.getLogger(__name__)

    logger.info("Computed equilibrium in %d iterations.", n_iter)
    logger.warning("Solver did not converge within %.2e.", target_tol)

Level guidelines:

- ``DEBUG``: detailed diagnostics, internal sub-solves (``suppress=True``), and
  messages gated by ``verbose=False``.
- ``INFO``: standard operational milestones and user-requested verbose/summary
  output.
- ``WARNING``: recoverable anomalies or non-convergence.
- ``ERROR``: failures (pass ``exc_info=True`` inside ``except`` blocks when a
  traceback is helpful).

Use :func:`warnings.warn` (not ``logger.warning``) for user-facing Python
warnings intended to be filtered via :mod:`warnings` or asserted with
``pytest.warns``.
