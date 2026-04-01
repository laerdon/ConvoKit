Decision Policy
===============

The decision policy API separates belief estimation (continuous scores) from
intervention decisions (discrete actions). This keeps ``Forecaster`` unchanged
while allowing flexible action logic in ``ForecasterModel``.

.. automodule:: convokit.decisionpolicy
    :members:
