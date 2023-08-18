# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class MetaReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class for handling many reports at the same time.

    :ivar reports: The many reports in a dictionary. For each report there
        must be an entry specifying "name" as string, "report" as a
        :class:`.Report` derived object, and "path_key" as the key of the
        argument used to write the report to a file.
    :vartype reports: dict
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, reports, **kwargs):
        """
        Root initialization for any instance of type Report.

        :param kwargs: The attributes for the report.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.reports = reports
        if reports is None:
            raise ReportException(
                'MetaReport with no reports is not supported.'
            )
        if not(
            isinstance(reports, list) or
            isinstance(reports, tuple) or
            isinstance(reports, np.ndarray)
        ):
            raise ReportException('MetaReport cannot handle given reports.')

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The concatenated string representation of the many reports.

        :return: String representation of the many reports.
        :rtype: str
        """
        str = ''
        for report in self.reports:
            str += report.to_string() + '\n\n----------\n\n'
        return str

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path, out_prefix=None, **kwargs):
        """
        Write the many reports to their corresponding files.

        :param path: The default path, i.e., the one associated with the
            "path" key.
        :param out_prefix: The output prefix to expand the path (OPTIONAL).
        :param kwargs: The key-word arguments. They must contain all relevant
            path specifications to handle each report.
        :return: Nothing, the output is written to the corresponding files.
        """
        # Handle path as kwargs path if no path is given in key-word arguments
        if kwargs.get('path', None) is None:
            kwargs['path'] = path
        # Each report to file
        start = time.perf_counter()
        for reporti in self.reports:
            name = reporti['name']
            report = reporti['report']
            outpath = kwargs[reporti['path_key']]
            report.to_file(outpath, out_prefix=out_prefix)
            LOGGING.LOGGER.debug(f'MetaReport handled "{name}" report.')
        # Log meta report end
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            f'MetaReport exported {len(self.reports)} reports to file in '
            f'{end-start:.3f} seconds.'
        )
