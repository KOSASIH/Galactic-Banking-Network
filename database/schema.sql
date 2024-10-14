-- Galactic Banking Network Database Schema

-- Users Table
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50) NOT NULL,
  password VARCHAR(255) NOT NULL,
  email VARCHAR(100) NOT NULL,
  role VARCHAR(50) NOT NULL, -- admin, user, etc.
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Accounts Table
CREATE TABLE accounts (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  account_number VARCHAR(50) NOT NULL,
  account_type VARCHAR(50) NOT NULL, -- checking, savings, etc.
  balance DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Transactions Table
CREATE TABLE transactions (
  id SERIAL PRIMARY KEY,
  account_id INTEGER NOT NULL,
  transaction_type VARCHAR(50) NOT NULL, -- deposit, withdrawal, transfer, etc.
  amount DECIMAL(10, 2) NOT NULL,
  transaction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (account_id) REFERENCES accounts(id)
);

-- Node Attributes Table
CREATE TABLE node_attributes (
  id SERIAL PRIMARY KEY,
  node_id INTEGER NOT NULL,
  attribute_name VARCHAR(50) NOT NULL,
  attribute_value VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Role Hierarchy Table
CREATE TABLE role_hierarchy (
  id SERIAL PRIMARY KEY,
  parent_role_id INTEGER,
  child_role_id INTEGER,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (parent_role_id) REFERENCES roles(id),
  FOREIGN KEY (child_role_id) REFERENCES roles(id)
);

-- Permission Matrix Table
CREATE TABLE permission_matrix (
  id SERIAL PRIMARY KEY,
  role_id INTEGER NOT NULL,
  permission_name VARCHAR(50) NOT NULL,
  permission_value BOOLEAN NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (role_id) REFERENCES roles(id)
);

-- Incident Response Table
CREATE TABLE incident_response (
  id SERIAL PRIMARY KEY,
  incident_id INTEGER NOT NULL,
  response_type VARCHAR(50) NOT NULL, -- alert, analysis, automated response, etc.
  response_data VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (incident_id) REFERENCES incidents(id)
);

-- Incidents Table
CREATE TABLE incidents (
  id SERIAL PRIMARY KEY,
  incident_type VARCHAR(50) NOT NULL, -- security breach, system failure, etc.
  incident_description VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- RSA Signatures Table
CREATE TABLE rsa_signatures (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  signature VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Permission Tokens Table
CREATE TABLE permission_tokens (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  token VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);
