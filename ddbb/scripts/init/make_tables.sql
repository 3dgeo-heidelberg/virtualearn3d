-- Author: Alberto M. Esmoris Pena
-- Brief: Script to make the tables for the catadb database
-- Database: catadb

-- TABLE: models
DROP TABLE IF EXISTS models cascade;
CREATE TABLE models (
    id bigserial PRIMARY KEY,
    predecessor_id BIGINT,
    model_type_id BIGINT NOT NULL,
    framework_id BIGINT NOT NULL,
    model_bin bytea,
    notes TEXT,
    foreign key(predecessor_id) references models(id)
);

-- TABLE: model_types
DROP TABLE IF EXISTS model_types cascade;
CREATE TABLE model_types (
    id bigserial PRIMARY KEY,
    specification JSONB NOT NULL,
    family_id INT NOT NULL,
    subfamily_id INT,
    notes TEXT
);
ALTER TABLE models ADD CONSTRAINT fk_model_types
    foreign key(model_type_id) REFERENCES model_types(id);

-- TABLE: model_families
DROP TABLE IF EXISTS model_families cascade;
CREATE TABLE model_families (
    id serial PRIMARY KEY,
    name VARCHAR(60) NOT NULL,
    description TEXT
);
ALTER TABLE model_types ADD CONSTRAINT fk_model_families
    foreign key(family_id) REFERENCES model_families(id);

-- TABLE: model_subfamilies
DROP TABLE IF EXISTS model_subfamilies cascade;
CREATE TABLE model_subfamilies (
    id serial PRIMARY KEY,
    name VARCHAR(60) NOT NULL,
    description TEXT
);
ALTER TABLE model_types ADD CONSTRAINT fk_model_subfamilies
    foreign key(subfamily_id) REFERENCES model_subfamilies(id);

-- TABLE: frameworks
DROP TABLE IF EXISTS frameworks cascade;
CREATE TABLE frameworks (
    id bigserial PRIMARY KEY,
    framework_name_id INT NOT NULL,
    version VARCHAR(90) NOT NULL,
    notes TEXT

);
ALTER TABLE models ADD CONSTRAINT fk_frameworks
    foreign key(framework_id) REFERENCES frameworks(id);

-- TABLE: framework_names
DROP TABLE IF EXISTS framework_names cascade;
CREATE TABLE framework_names (
    id serial PRIMARY KEY,
    name VARCHAR(60) NOT NULL,
    description TEXT
);
ALTER TABLE frameworks ADD CONSTRAINT fk_framework_names
    foreign key(framework_name_id) REFERENCES framework_names(id);

-- TABLE: modelers
DROP TABLE IF EXISTS modelers cascade;
CREATE TABLE modelers (
    id bigserial PRIMARY KEY,
    name VARCHAR(60) UNIQUE NOT NULL,
    description TEXT,
    email VARCHAR(256),
    phone VARCHAR(60),
    url TEXT
);

-- TABLE: model_modelers
DROP TABLE IF EXISTS model_modelers cascade;
CREATE TABLE model_modelers (
    model_id BIGINT NOT NULL,
    modeler_id BIGINT NOT NULL,
    foreign key(model_id) REFERENCES models(id),
    foreign key(modeler_id) REFERENCES modelers(id),
    PRIMARY KEY(model_id, modeler_id)
);

-- TABLE: projects
DROP TABLE IF EXISTS projects cascade;
CREATE TABLE projects (
    id bigserial PRIMARY KEY,
    name VARCHAR(256) UNIQUE NOT NULL,
    description TEXT
);

-- TABLE: project_modelers
DROP TABLE IF EXISTS project_modelers cascade;
CREATE TABLE project_modelers (
    project_id BIGINT NOT NULL,
    modeler_id BIGINT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id),
    FOREIGN KEY(modeler_id) REFERENCES modelers(id),
    PRIMARY KEY(project_id, modeler_id)
);

-- TABLE: project_models
DROP TABLE IF EXISTS project_models cascade;
CREATE TABLE project_models (
    project_id BIGINT NOT NULL,
    model_id BIGINT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id),
    FOREIGN KEY(model_id) REFERENCES models(id),
    PRIMARY KEY(project_id, model_id)
);

-- TABLE: data_domains
DROP TABLE IF EXISTS data_domains cascade;
CREATE TABLE data_domains (
    id serial PRIMARY KEY,
    name VARCHAR(60),
    description TEXT
);

-- TABLE: target_domains
DROP TABLE IF EXISTS target_domains cascade;
CREATE TABLE target_domains (
    id serial PRIMARY KEY,
    name VARCHAR(60),
    description TEXT
);

-- TABLE: domains
DROP TABLE IF EXISTS domains cascade;
CREATE TABLE domains(
    id serial PRIMARY KEY,
    data_domain_id INT NOT NULL,
    target_domain_id INT NOT NULL,
    FOREIGN KEY(data_domain_id) REFERENCES data_domains(id),
    FOREIGN KEY(target_domain_id) REFERENCES target_domains(id)
);

-- TABLE: project_domains
DROP TABLE IF EXISTS project_domains cascade;
CREATE TABLE project_domains(
    project_id BIGINT NOT NULL,
    domain_id INT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id),
    FOREIGN KEY(domain_id) REFERENCES domains(id),
    PRIMARY KEY(project_id, domain_id)
);

-- TABLE: tasks
DROP TABLE IF EXISTS tasks cascade;
CREATE TABLE tasks(
    id serial PRIMARY KEY,
    type VARCHAR(60) NOT NULL,
    description TEXT
);

-- TABLE: model_tasks
DROP TABLE IF EXISTS model_tasks cascade;
CREATE TABLE model_tasks(
    model_type_id BIGINT NOT NULL,
    task_id INT NOT NULL,
    FOREIGN KEY(model_type_id) REFERENCES model_types(id),
    FOREIGN KEY(task_id) REFERENCES tasks(id),
    PRIMARY KEY(model_type_id, task_id)
);

-- TABLE: model_domains
DROP TABLE IF EXISTS model_domains cascade;
CREATE TABLE model_domains(
    model_type_id BIGINT NOT NULL,
    domain_id INT NOT NULL,
    FOREIGN KEY(model_type_id) REFERENCES model_types(id),
    FOREIGN KEY(domain_id) REFERENCES domains(id),
    PRIMARY KEY(model_type_id, domain_id)
);

-- TABLE: training_metrics
DROP TABLE IF EXISTS training_metrics cascade;
CREATE TABLE training_metrics(
    id serial PRIMARY KEY,
    name VARCHAR(60) NOT NULL,
    description TEXT
);

-- TABLE: training_histories
DROP TABLE IF EXISTS training_histories cascade;
CREATE TABLE training_histories(
    model_id BIGINT NOT NULL,
    metric_id INT NOT NULL,
    num_epochs INT NOT NULL,
    start_value FLOAT NOT NULL,
    end_value FLOAT NOT NULL,
    min_value FLOAT NOT NULL,
    min_value_epoch INT NOT NULL,
    max_value FLOAT NOT NULL,
    max_value_epoch INT NOT NULL,
    mean_value FLOAT NOT NULL,
    stdev_value FLOAT NOT NULL,
    notes TEXT,
    FOREIGN KEY(model_id) REFERENCES models(id),
    FOREIGN KEY(metric_id) REFERENCES training_metrics(id),
    PRIMARY KEY(model_id, metric_id)
);

-- TABLE: datasets
DROP TABLE IF EXISTS datasets cascade;
CREATE TABLE datasets(
    id bigserial PRIMARY KEY,
    name VARCHAR(60) NOT NULL,
    num_points BIGINT NOT NULL,
    num_references BIGINT,
    pxmin FLOAT,
    pxmax FLOAT,
    pymin FLOAT,
    pymax FLOAT,
    pzmin FLOAT,
    pzmax FLOAT,
    notes TEXT
);

-- TABLE: dataset_domains
DROP TABLE IF EXISTS dataset_domains cascade;
CREATE TABLE dataset_domains(
    dataset_id INT NOT NULL,
    domain_id INT NOT NULL,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id),
    FOREIGN KEY(domain_id) REFERENCES domains(id),
    PRIMARY KEY(dataset_id, domain_id)
);

-- TABLE: metadatasets
DROP TABLE IF EXISTS metadatasets cascade;
CREATE TABLE metadatasets(
    id serial PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    url TEXT,
    contact TEXT,
    owner TEXT,
    open_access BOOL,
    unrestricted_open_access BOOL
);

-- TABLE: dataset_metas
DROP TABLE IF EXISTS dataset_metas cascade;
CREATE TABLE dataset_metas (
    dataset_id BIGINT NOT NULL,
    meta_id INT NOT NULL,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id),
    FOREIGN KEY(meta_id) REFERENCES metadatasets(id),
    PRIMARY KEY(dataset_id, meta_id)
);

-- TABLE: machines
DROP TABLE IF EXISTS machines cascade;
CREATE TABLE machines (
    id serial PRIMARY KEY,
    name VARCHAR(60) UNIQUE,
    cpu TEXT,
    cpu_max_freq INT,
    cpu_max_cores INT,
    gpu TEXT,
    gpu_max_freq INT,
    gpu_max_cores INT,
    gpu_max_mem BIGINT,
    tpu TEXT,
    tpu_max_freq INT,
    tpu_max_cores INT,
    tpu_max_mem BIGINT,
    ram TEXT,
    ram_max_mem BIGINT,
    notes TEXT
);

-- TABLE: executors
DROP TABLE IF EXISTS executors cascade;
CREATE TABLE executors(
    id bigserial PRIMARY KEY,
    machine_id INT NOT NULL,
    num_cpu_cores INT NOT NULL,
    num_cpus INT NOT NULL,
    num_tpus INT NOT NULL,
    ram_mem BIGINT NOT NULL,
    notes TEXT
);

-- TABLE: model_runtimes
DROP TABLE IF EXISTS model_runtimes cascade;
CREATE TABLE model_runtimes(
    model_id BIGINT NOT NULL,
    dataset_id BIGINT NOT NULL,
    executor_id BIGINT NOT NULL,
    texec_mean FLOAT NOT NULL,
    texec_stdev FLOAT,
    texec_q1 FLOAT,
    texec_q2 FLOAT,
    texec_q3 FLOAT,
    notes TEXT
);

-- TABLE: plots
DROP TABLE IF EXISTS plots cascade;
CREATE TABLE plots(
    id serial PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT
);

-- TABLE: plot_formats
DROP TABLE IF EXISTS plot_formats cascade;
CREATE TABLE plot_formats(
    id serial PRIMARY KEY,
    name VARCHAR(60) NOT NULL UNIQUE,
    description TEXT
);

-- TABLE: model_plots
DROP TABLE IF EXISTS model_plots cascade;
CREATE TABLE model_plots(
    model_id BIGINT NOT NULL,
    plot_id INT NOT NULL,
    plot_bin bytea NOT NULL,
    plot_format_id INT NOT NULL,
    notes TEXT,
    FOREIGN KEY(model_id) REFERENCES models(id),
    FOREIGN KEY(plot_id) REFERENCES plots(id),
    PRIMARY KEY(model_id, plot_id)
);

-- TABLE: resultsets
DROP TABLE IF EXISTS resultsets cascade;
CREATE TABLE resultsets(
    id bigserial PRIMARY KEY,
    model_id BIGINT NOT NULL,
    dataset_id BIGINT NOT NULL,
    global_resultset_id BIGINT,
    result_date TIMESTAMP NOT NULL DEFAULT now(),
    notes TEXT,
    FOREIGN KEY(model_id) REFERENCES models(id),
    FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);

-- TABLE: global_resultsets
DROP TABLE IF EXISTS global_resultsets cascade;
CREATE TABLE global_resultsets(
    id bigserial PRIMARY KEY,
    oa FLOAT,
    p FLOAT,
    r FLOAT,
    f1 FLOAT,
    iou FLOAT,
    wp FLOAT,
    wr FLOAT,
    wf1 FLOAT,
    wiou FLOAT,
    mcc FLOAT,
    kappa FLOAT,
    notes TEXT
);
ALTER TABLE resultsets ADD CONSTRAINT fk_global_resultsets
    FOREIGN KEY(global_resultset_id) REFERENCES global_resultsets(id);

-- TABLE: regression_metrics
DROP TABLE IF EXISTS regression_metrics cascade;
CREATE TABLE regression_metrics(
    id serial PRIMARY KEY,
    name VARCHAR(60) NOT NULL,
    description TEXT
);

-- TABLE: regression_resultsets
DROP TABLE IF EXISTS regression_resultsets cascade;
CREATE TABLE regression_resultsets(
    resultset_id BIGINT NOT NULL,
    metric_id INT NOT NULL,
    mean FLOAT,
    stdev FLOAT,
    q1 FLOAT,
    q2 FLOAT,
    q3 FLOAT,
    q4 FLOAT,
    q5 FLOAT,
    q6 FLOAT,
    q7 FLOAT,
    q8 FLOAT,
    q9 FLOAT,
    r2 FLOAT,
    rmse FLOAT,
    rse_stdev FLOAT,
    mae FLOAT,
    ae_stdev FLOAT,
    notes TEXT,
    FOREIGN KEY(resultset_id) REFERENCES resultsets(id),
    FOREIGN KEY(metric_id) REFERENCES regression_metrics(id),
    PRIMARY KEY(resultset_id, metric_id)
);

-- TABLE: regression_references
DROP TABLE IF EXISTS regression_references cascade;
CREATE TABLE regression_references(
    dataset_id BIGINT NOT NULL,
    metric_id INT NOT NULL,
    mean FLOAT,
    stdev FLOAT,
    q1 FLOAT,
    q2 FLOAT,
    q3 FLOAT,
    q4 FLOAT,
    q5 FLOAT,
    q6 FLOAT,
    q7 FLOAT,
    q8 FLOAT,
    q9 FLOAT,
    notes TEXT,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id),
    FOREIGN KEY(metric_id) REFERENCES regression_metrics(id),
    PRIMARY KEY(dataset_id, metric_id)
);

-- TABLE: uncertainty_metrics
DROP TABLE IF EXISTS uncertainty_metrics cascade;
CREATE TABLE uncertainty_metrics(
    id SERIAL PRIMARY KEY,
    name VARCHAR(60) NOT NULL,
    description TEXT
);

-- TABLE: uncertainty_resultsets
DROP TABLE IF EXISTS uncertainty_resultsets cascade;
CREATE TABLE uncertainty_resultsets(
    resultset_id BIGINT NOT NULL,
    metric_id INT NOT NULL,
    mean FLOAT,
    stdev FLOAT,
    q1 FLOAT,
    q2 FLOAT,
    q3 FLOAT,
    q4 FLOAT,
    q5 FLOAT,
    q6 FLOAT,
    q7 FLOAT,
    q8 FLOAT,
    q9 FLOAT,
    notes TEXT
);

-- TABLE: classes
DROP TABLE IF EXISTS classes cascade;
CREATE TABLE classes(
    id serial PRIMARY KEY,
    name VARCHAR(60) NOT NULL,
    description TEXT
);

-- TABLE: classwise_resultsets
DROP TABLE IF EXISTS classwise_resultsets cascade;
CREATE TABLE classwise_resultsets(
    resultset_id BIGINT NOT NULL,
    class_id INT NOT NULL,
    p FLOAT,
    r FLOAT,
    f1 FLOAT,
    iou FLOAT,
    mcc FLOAT,
    kappa FLOAT,
    notes TEXT,
    FOREIGN KEY(resultset_id) REFERENCES resultsets(id),
    FOREIGN KEY(class_id) REFERENCES classes(id),
    PRIMARY KEY(resultset_id, class_id)
);

-- TABLE: class_distributions
DROP TABLE IF EXISTS classwise_distributions cascade;
CREATE TABLE classwise_distributions(
    dataset_id BIGINT NOT NULL,
    class_id INT NOT NULL,
    recount BIGINT NOT NULL,
    notes TEXT,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id),
    FOREIGN KEY(class_id) REFERENCES classes(id),
    PRIMARY KEY(dataset_id, class_id)
);
