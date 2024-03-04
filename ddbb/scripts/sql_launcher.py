# -----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Script to launch SQL remotely in the catadb PostgreSQL database
# -----------------------------------------------------------------------------

# ---   IMPORTS   --- #
# ------------------- #
import psycopg2
import json
import re
import sys
import os


# ---   HELP   --- #
# ---------------- #
def print_help():
    print(
'''
USAGE of sql_launcher.py:

    1: Path to the JSON file where connection data is specified.

    2..n: Path to a sequence of SQL scripts that must be launched.
'''
)

# ---   INPUT HANDLING   --- #
# -------------------------- #
def read_args():
    # Check enough arguments are given
    if len(sys.argv) < 3:
        print_help()
        sys.exit(1)
    # Validate JSON specification
    con_json = sys.argv[1]
    if not os.path.isfile(con_json):
        raise ValueError(
            'Wrong path to JSON file with connection data:\n'
            f'"{con_json}"'
        )
    with open(con_json, "r") as con_jsonf:
        con_json = json.load(con_jsonf)
    # Validate SQL scripts
    sqls = sys.argv[2:]
    for sql in sqls:
        if not os.path.isfile(sql):
            raise ValueError(
                f'Wrong path to SQL script file:\n"{json}"'
            )
    # Return JSON and SQL scripts
    return con_json, sqls

# ---   SQL HANDLING   --- #
# ------------------------ #
def launch_script(con, sql_script):
    """
    Launch a SQL script
    :param con: The SQL connection
    :param sql_script: Path to the SQL script
    """
    print(f'Launching "{sql_script}" ...')
    # Read SQL script
    with open(sql_script, 'r') as f:
        # REGEXP description
        # () IS the pattern wrapper
        # (?:) IS a non-capture group
        # [^;"']  IS everything but ;"'
        # "[^"]*" IS everything inside a pair of ""
        # '[^']*' IS everything inside a pair of ''
        # | IS or
        # + IS one or more
        sql = re.compile(
            r'''((?:[^;"']|"[^"]*"|'[^']*')+)'''
        ).split(f.read())[1:-1:2]
        cursor = con.cursor()
        for sub_sql in sql:
            try:
                with con.cursor() as cursor:
                    cursor.execute(sub_sql)
                    cursor.close()
            except Exception as ex:
                s = sub_sql.replace(' ', "").replace("\n", "")
                if len(s) > 0:
                    print(f'Exception: {ex}')
    print(f'Launched "{sql_script}"!\n------------------------\n\n')

# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    con_json, sql_scripts = read_args()  # Read input
    con = psycopg2.connect(**con_json)  # Connect to database
    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        for sql_script in sql_scripts:  # Launch scripts
                launch_script(con, sql_script)  # Launch the script itself
    finally:
        con.close()  # Close connection to database




