-- Author: Alberto M. Esmoris Pena
-- Brief: catadb is a database for model analytics in the context of the VL3D
--          framework.
-- Database: catadb



-- Drop previous catadb, if any
DROP DATABASE IF EXISTS catadb;

-- Create new catadb
CREATE DATABASE catadb
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    LOCALE_PROVIDER = 'libc'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;

COMMENT ON DATABASE catadb
    IS 'database for VL3D model analytics';



