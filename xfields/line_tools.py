# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

from .beam_elements.touschek import TouschekScattering
from .touschek.study import TouschekStudy


class XfieldsLineAPI:
    """
    Xfields-specific helpers associated with an xtrack Line.

    The public methods use flat prefixes (for example ``touschek_*``) so the
    reference guide can document one facade class without introducing many
    nested API containers.
    """

    def __init__(self, line):
        self._line = line

    @property
    def line(self):
        return self._line

    def _touschek_elements(self, elements=None):
        line = self.line

        if elements is None:
            tab = line.get_table()
            elements = [
                nn for nn in tab.name[:-1]
                if isinstance(line[nn], TouschekScattering)
            ]
        elif isinstance(elements, str):
            elements = [elements]
        else:
            elements = list(elements)

        if len(elements) == 0:
            raise ValueError(
                "No TouschekScattering elements found. Add them to the line "
                "before using the Touschek facade."
            )

        for nn in elements:
            if nn not in line.element_names:
                raise ValueError(f"Element '{nn}' is not present in the line.")
            if not isinstance(line[nn], TouschekScattering):
                raise TypeError(
                    f"Element '{nn}' is not a TouschekScattering "
                    f"(got {type(line[nn]).__name__})."
                )

        return elements

    def touschek_configure(self, *, local_momentum_acceptance, twiss=None,
                           elements=None, **kwargs):
        """
        Configure Touschek scattering elements in the line.

        Returns a configured :class:`xfields.TouschekStudy`, on which users can
        call ``local_rates()``, ``generate_particles()``, or ``run()``.
        """
        elements = self._touschek_elements(elements)

        study = TouschekStudy(
            self.line,
            elements=elements,
            twiss=twiss,
            local_momentum_acceptance=local_momentum_acceptance,
            **kwargs,
        )
        study.initialise_touschek()
        return study

    def touschek_run(self, *, local_momentum_acceptance, twiss=None,
                     elements=None, track=False, n_turns=None,
                     generate_particles=None, with_progress=False, **kwargs):
        """
        Configure and run a Touschek study.

        This is a shortcut for simple scripts. For clearer staged workflows,
        use ``study = line.xfields.touschek_configure(...)`` followed by
        ``study.run(...)``.
        """
        study = self.touschek_configure(
            local_momentum_acceptance=local_momentum_acceptance,
            twiss=twiss,
            elements=elements,
            **kwargs,
        )

        return study.run(
            track=track,
            n_turns=n_turns,
            generate_particles=generate_particles,
            with_progress=with_progress,
        )
