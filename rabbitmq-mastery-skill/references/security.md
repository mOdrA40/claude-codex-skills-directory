# RabbitMQ Security Hardening Guide

## Authentication

### 1. User Management

```bash
# NEVER use default guest user in production
rabbitmqctl delete_user guest

# Create application users with minimal permissions
rabbitmqctl add_user order_service "$(openssl rand -base64 32)"
rabbitmqctl add_user payment_service "$(openssl rand -base64 32)"
rabbitmqctl add_user monitoring "$(openssl rand -base64 32)"

# Admin user (use sparingly)
rabbitmqctl add_user admin "$(openssl rand -base64 32)"
rabbitmqctl set_user_tags admin administrator
```

### 2. Permission Model

```bash
# Permission format: configure, write, read (regex patterns)

# Order service: full access to order.* queues/exchanges
rabbitmqctl set_permissions -p orders order_service \
    "^order\\..*" "^order\\..*" "^order\\..*"

# Payment service: write to payment.*, read from order.*
rabbitmqctl set_permissions -p orders payment_service \
    "" "^payment\\..*" "^(payment|order)\\..*"

# Monitoring: read-only everywhere
rabbitmqctl set_permissions -p orders monitoring \
    "" "" ".*"

# Principle of least privilege matrix
# | User            | Configure | Write    | Read     |
# |-----------------|-----------|----------|----------|
# | order_service   | order.*   | order.*  | order.*  |
# | payment_service | (none)    | payment.*| payment.*,order.* |
# | monitoring      | (none)    | (none)   | .*       |
```

### 3. Virtual Host Isolation

```bash
# Create vhosts per environment/tenant
rabbitmqctl add_vhost production
rabbitmqctl add_vhost staging
rabbitmqctl add_vhost tenant_acme
rabbitmqctl add_vhost tenant_globex

# Users can only access their vhost
rabbitmqctl set_permissions -p tenant_acme acme_app ".*" ".*" ".*"
# No permissions on other vhosts = cannot access
```

## TLS/SSL Configuration

### 1. Generate Certificates

```bash
#!/bin/bash
# generate_certs.sh

DOMAIN="rabbitmq.company.com"
DAYS=365
CA_DIR="./ca"
SERVER_DIR="./server"
CLIENT_DIR="./client"

mkdir -p $CA_DIR $SERVER_DIR $CLIENT_DIR

# Generate CA
openssl genrsa -out $CA_DIR/ca.key 4096
openssl req -new -x509 -days 3650 -key $CA_DIR/ca.key \
    -out $CA_DIR/ca.crt \
    -subj "/CN=RabbitMQ-CA/O=Company/C=US"

# Generate Server Certificate
openssl genrsa -out $SERVER_DIR/server.key 4096
openssl req -new -key $SERVER_DIR/server.key \
    -out $SERVER_DIR/server.csr \
    -subj "/CN=$DOMAIN/O=Company/C=US"

# Server cert extensions
cat > $SERVER_DIR/server.ext << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names
[alt_names]
DNS.1 = $DOMAIN
DNS.2 = rabbitmq-1.$DOMAIN
DNS.3 = rabbitmq-2.$DOMAIN
DNS.4 = rabbitmq-3.$DOMAIN
DNS.5 = localhost
IP.1 = 127.0.0.1
EOF

openssl x509 -req -in $SERVER_DIR/server.csr \
    -CA $CA_DIR/ca.crt -CAkey $CA_DIR/ca.key -CAcreateserial \
    -out $SERVER_DIR/server.crt -days $DAYS \
    -extfile $SERVER_DIR/server.ext

# Generate Client Certificate (for mTLS)
openssl genrsa -out $CLIENT_DIR/client.key 4096
openssl req -new -key $CLIENT_DIR/client.key \
    -out $CLIENT_DIR/client.csr \
    -subj "/CN=order_service/O=Company/C=US"

openssl x509 -req -in $CLIENT_DIR/client.csr \
    -CA $CA_DIR/ca.crt -CAkey $CA_DIR/ca.key -CAcreateserial \
    -out $CLIENT_DIR/client.crt -days $DAYS

# Set permissions
chmod 600 $SERVER_DIR/server.key $CLIENT_DIR/client.key
chmod 644 $CA_DIR/ca.crt $SERVER_DIR/server.crt $CLIENT_DIR/client.crt

echo "Certificates generated successfully"
```

### 2. Server TLS Configuration

```ini
# /etc/rabbitmq/rabbitmq.conf

# Disable non-TLS listeners in production
listeners.tcp = none

# TLS listener
listeners.ssl.default = 5671

# Certificate paths
ssl_options.cacertfile = /etc/rabbitmq/certs/ca.crt
ssl_options.certfile = /etc/rabbitmq/certs/server.crt
ssl_options.keyfile = /etc/rabbitmq/certs/server.key

# TLS version (TLS 1.2+ only)
ssl_options.versions.1 = tlsv1.3
ssl_options.versions.2 = tlsv1.2

# Strong cipher suites only
ssl_options.ciphers.1 = TLS_AES_256_GCM_SHA384
ssl_options.ciphers.2 = TLS_AES_128_GCM_SHA256
ssl_options.ciphers.3 = TLS_CHACHA20_POLY1305_SHA256
ssl_options.ciphers.4 = ECDHE-RSA-AES256-GCM-SHA384
ssl_options.ciphers.5 = ECDHE-RSA-AES128-GCM-SHA256

# Require client certificate (mTLS)
ssl_options.verify = verify_peer
ssl_options.fail_if_no_peer_cert = true

# Honor server cipher order
ssl_options.honor_cipher_order = true
ssl_options.honor_ecc_order = true

# Management TLS
management.ssl.port = 15671
management.ssl.cacertfile = /etc/rabbitmq/certs/ca.crt
management.ssl.certfile = /etc/rabbitmq/certs/server.crt
management.ssl.keyfile = /etc/rabbitmq/certs/server.key
management.ssl.versions.1 = tlsv1.3
management.ssl.versions.2 = tlsv1.2
```

### 3. Client TLS Connection

```python
import ssl
import pika

# Create SSL context
context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
context.load_verify_locations('/path/to/ca.crt')

# For mTLS (client certificate)
context.load_cert_chain(
    certfile='/path/to/client.crt',
    keyfile='/path/to/client.key'
)

# Strict verification
context.check_hostname = True
context.verify_mode = ssl.CERT_REQUIRED

# Minimum TLS version
context.minimum_version = ssl.TLSVersion.TLSv1_2

ssl_options = pika.SSLOptions(context, 'rabbitmq.company.com')

parameters = pika.ConnectionParameters(
    host='rabbitmq.company.com',
    port=5671,
    credentials=pika.PlainCredentials('order_service', 'password'),
    ssl_options=ssl_options,
    heartbeat=60
)

connection = pika.BlockingConnection(parameters)
```

## OAuth 2.0 / JWT Authentication

### 1. Enable OAuth Plugin

```bash
rabbitmq-plugins enable rabbitmq_auth_backend_oauth2
```

### 2. OAuth Configuration

```ini
# rabbitmq.conf

# Authentication backends (OAuth first, then internal)
auth_backends.1 = rabbit_auth_backend_oauth2
auth_backends.2 = rabbit_auth_backend_internal

# OAuth2 / JWT settings
auth_oauth2.resource_server_id = rabbitmq
auth_oauth2.issuer = https://auth.company.com
auth_oauth2.https.cacertfile = /etc/rabbitmq/certs/oauth-ca.crt

# JWKS endpoint for key validation
auth_oauth2.jwks_url = https://auth.company.com/.well-known/jwks.json

# Token validation
auth_oauth2.preferred_username_claims.1 = sub
auth_oauth2.preferred_username_claims.2 = client_id

# Scope to permission mapping
auth_oauth2.scope_prefix = rabbitmq.
auth_oauth2.additional_scopes_key = permissions
```

### 3. JWT Token Structure

```json
{
  "sub": "order_service",
  "aud": "rabbitmq",
  "iss": "https://auth.company.com",
  "exp": 1735689600,
  "iat": 1735603200,
  "scope": "rabbitmq.read:*/orders/* rabbitmq.write:*/orders/* rabbitmq.configure:*/orders/*"
}
```

### 4. Client with OAuth

```python
import requests
import pika

def get_oauth_token():
    response = requests.post(
        'https://auth.company.com/oauth/token',
        data={
            'grant_type': 'client_credentials',
            'client_id': 'order_service',
            'client_secret': 'secret',
            'scope': 'rabbitmq.read:*/orders/* rabbitmq.write:*/orders/*'
        }
    )
    return response.json()['access_token']

token = get_oauth_token()

# Use token as password, empty username
credentials = pika.PlainCredentials('', token)

parameters = pika.ConnectionParameters(
    host='rabbitmq.company.com',
    port=5671,
    credentials=credentials,
    ssl_options=ssl_options
)
```

## LDAP Integration

```ini
# rabbitmq.conf

auth_backends.1 = rabbit_auth_backend_ldap
auth_backends.2 = rabbit_auth_backend_internal

# LDAP server
auth_ldap.servers.1 = ldap.company.com
auth_ldap.servers.2 = ldap2.company.com
auth_ldap.port = 636
auth_ldap.use_ssl = true
auth_ldap.ssl_options.cacertfile = /etc/rabbitmq/certs/ldap-ca.crt

# Service account for LDAP queries
auth_ldap.dn_lookup_bind.user_dn = cn=rabbitmq,ou=services,dc=company,dc=com
auth_ldap.dn_lookup_bind.password = service_password

# User lookup
auth_ldap.dn_lookup_attribute = sAMAccountName
auth_ldap.dn_lookup_base = ou=users,dc=company,dc=com

# Group-based authorization
auth_ldap.vhost_access_query.1.match.who.s = member
auth_ldap.vhost_access_query.1.match.what.s = cn=rabbitmq-users
auth_ldap.vhost_access_query.1.match.against = "cn=*,ou=groups,dc=company,dc=com"

auth_ldap.resource_access_query.1.match.who.s = member
auth_ldap.resource_access_query.1.match.what.s = cn=rabbitmq-admins
auth_ldap.resource_access_query.1.return = true

auth_ldap.tag_queries.1.who.s = member
auth_ldap.tag_queries.1.what.s = cn=rabbitmq-admins
auth_ldap.tag_queries.1.tags.1 = administrator
```

## Network Security

### 1. Firewall Rules

```bash
# iptables rules for RabbitMQ cluster

# AMQP (TLS only)
iptables -A INPUT -p tcp --dport 5671 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 5671 -j DROP

# Management (internal only)
iptables -A INPUT -p tcp --dport 15671 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 15671 -j DROP

# Erlang distribution (cluster nodes only)
iptables -A INPUT -p tcp --dport 25672 -s 10.0.1.10 -j ACCEPT
iptables -A INPUT -p tcp --dport 25672 -s 10.0.1.11 -j ACCEPT
iptables -A INPUT -p tcp --dport 25672 -s 10.0.1.12 -j ACCEPT
iptables -A INPUT -p tcp --dport 25672 -j DROP

# Erlang epmd (cluster nodes only)
iptables -A INPUT -p tcp --dport 4369 -s 10.0.1.10 -j ACCEPT
iptables -A INPUT -p tcp --dport 4369 -s 10.0.1.11 -j ACCEPT
iptables -A INPUT -p tcp --dport 4369 -s 10.0.1.12 -j ACCEPT
iptables -A INPUT -p tcp --dport 4369 -j DROP

# Prometheus metrics (monitoring only)
iptables -A INPUT -p tcp --dport 15692 -s 10.0.2.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 15692 -j DROP
```

### 2. Inter-node Encryption

```ini
# rabbitmq.conf

# Encrypt Erlang distribution (inter-node traffic)
ssl_options.cacertfile = /etc/rabbitmq/certs/ca.crt
ssl_options.certfile = /etc/rabbitmq/certs/server.crt
ssl_options.keyfile = /etc/rabbitmq/certs/server.key

# Enable distribution TLS
cluster_formation.ssl.cacertfile = /etc/rabbitmq/certs/ca.crt
cluster_formation.ssl.certfile = /etc/rabbitmq/certs/server.crt
cluster_formation.ssl.keyfile = /etc/rabbitmq/certs/server.key
cluster_formation.ssl.verify = verify_peer
cluster_formation.ssl.fail_if_no_peer_cert = true
```

```erlang
%% /etc/rabbitmq/advanced.config
[
    {rabbit, [
        {ssl_options, [
            {cacertfile, "/etc/rabbitmq/certs/ca.crt"},
            {certfile, "/etc/rabbitmq/certs/server.crt"},
            {keyfile, "/etc/rabbitmq/certs/server.key"}
        ]}
    ]},
    {kernel, [
        {inet_dist_use_interface, {0,0,0,0}},
        {net_ticktime, 60}
    ]},
    {ssl, [
        {versions, ['tlsv1.3', 'tlsv1.2']}
    ]}
].
```

## Audit Logging

### 1. Enable Audit Logging

```ini
# rabbitmq.conf

# File-based audit log
log.file = /var/log/rabbitmq/rabbit.log
log.file.level = info

# Separate audit log
log.file.1 = /var/log/rabbitmq/audit.log
log.file.1.level = info
log.file.1.formatter = json

# Log rotation
log.file.rotation.date = $D0
log.file.rotation.count = 30
log.file.rotation.size = 104857600

# Events to log
log.connection.level = info
log.channel.level = info
log.queue.level = info
log.mirroring.level = info
log.federation.level = info
log.shovel.level = info
```

### 2. Syslog Integration

```ini
# rabbitmq.conf

log.syslog = true
log.syslog.transport = tcp
log.syslog.host = syslog.company.com
log.syslog.port = 514
log.syslog.facility = daemon
log.syslog.level = info
```

### 3. Audit Events to Monitor

```yaml
# Security-relevant events
- Connection opened/closed (who, when, from where)
- Authentication failures
- Authorization failures (permission denied)
- User/permission changes
- Queue/exchange declarations
- Policy changes
- Cluster membership changes
- Plugin enable/disable
```

## Security Checklist

```markdown
# Production Security Checklist

## Authentication
- [ ] Default guest user deleted
- [ ] Strong passwords (32+ char random)
- [ ] Password rotation policy in place
- [ ] Service accounts use minimal permissions
- [ ] OAuth2/mTLS for service-to-service auth

## Network
- [ ] Non-TLS ports disabled (5672)
- [ ] TLS 1.2+ only
- [ ] Strong cipher suites only
- [ ] Inter-node traffic encrypted
- [ ] Firewall rules configured
- [ ] Management UI not exposed publicly
- [ ] VPN/private network for cluster traffic

## Authorization
- [ ] Vhost isolation per environment/tenant
- [ ] Least privilege permissions
- [ ] No wildcard permissions in production
- [ ] Regular permission audits

## Monitoring
- [ ] Audit logging enabled
- [ ] Failed auth attempts alerted
- [ ] Unusual traffic patterns detected
- [ ] Certificate expiry monitored

## Secrets Management
- [ ] Credentials in vault (HashiCorp, AWS SM)
- [ ] No hardcoded passwords
- [ ] Erlang cookie rotated periodically
- [ ] Certificate auto-renewal (cert-manager)
```
