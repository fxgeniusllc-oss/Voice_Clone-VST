# Dynamic Firewall Configuration for GitHub Actions

This document describes the dynamic firewall configuration system implemented for the MAEVN CI/CD pipeline. The system automatically configures firewall rules to whitelist GitHub Actions runner IP addresses before executing build and test jobs.

## Overview

GitHub Actions runners use dynamic IP addresses that change with each workflow run. When your CI/CD pipeline needs to access resources protected by a firewall (such as private package repositories, databases, or internal APIs), you need to dynamically whitelist the runner's IP address.

This implementation provides a **pre-rules-based** approach where firewall rules are configured **before** the main build jobs execute, ensuring that all protected resources are accessible throughout the workflow.

## Architecture

### Workflow Structure

```
┌─────────────────────────────────────────────────────────┐
│  1. Configure Firewall (Pre-Rules)                      │
│     - Detect runner IP address                          │
│     - Create firewall allow rule for runner IP          │
│     - Verify connectivity                               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  2. Build Jobs (Parallel)                               │
│     - Build on Linux                                     │
│     - Build on macOS                                     │
│     - Build on Windows                                   │
│     - All jobs can access protected resources           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  3. Cleanup Firewall (Post-Rules)                       │
│     - Remove temporary firewall rules                   │
│     - Clean up resources                                │
│     - Always runs (even if builds fail)                 │
└─────────────────────────────────────────────────────────┘
```

## Features

### 1. Dynamic IP Detection

The system automatically detects the GitHub Actions runner's public IP address:

```yaml
- name: Get Runner IP Address
  id: get-ip
  run: |
    RUNNER_IP=$(curl -s https://api.ipify.org)
    echo "ip=$RUNNER_IP" >> $GITHUB_OUTPUT
```

**Reliability:** Uses multiple IP detection services as fallback.

### 2. Multi-Provider Support

The implementation supports multiple firewall providers:

#### AWS Security Groups
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 443 \
  --cidr ${RUNNER_IP}/32 \
  --description "GitHub Actions Runner - Job ${RUN_ID}"
```

#### Azure Network Security Groups (NSG)
```bash
az network nsg rule create \
  --resource-group ${RESOURCE_GROUP} \
  --nsg-name ${NSG_NAME} \
  --name "GitHub-Actions-${RUN_ID}" \
  --priority 1000 \
  --source-address-prefixes "${RUNNER_IP}/32" \
  --destination-port-ranges 443 \
  --access Allow
```

#### Google Cloud Platform (GCP)
```bash
gcloud compute firewall-rules create github-actions-${RUN_ID} \
  --network=${NETWORK} \
  --action=ALLOW \
  --rules=tcp:443 \
  --source-ranges="${RUNNER_IP}/32"
```

#### Custom Firewall API
```bash
curl -X POST https://api.firewall.example.com/rules \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"ip": "${RUNNER_IP}", "action": "allow"}'
```

### 3. Automatic Cleanup

Firewall rules are automatically removed after jobs complete:

```yaml
cleanup-firewall:
  needs: [configure-firewall, build-linux, build-macos, build-windows]
  if: always()  # Runs even if builds fail
```

### 4. Job Dependencies

Build jobs depend on firewall configuration:

```yaml
build-linux:
  needs: configure-firewall
  # Only runs after firewall is configured
```

## Usage

### Basic CI Workflow

The `ci.yml` workflow automatically runs on:
- Push to `main`, `dev`, or `copilot/**` branches
- Pull requests to `main` or `dev`
- Manual workflow dispatch

```bash
# Workflow runs automatically on push
git push origin main

# Or trigger manually via GitHub UI
# Actions → CI Build and Test → Run workflow
```

### Advanced Firewall Configuration

The `firewall-config.yml` workflow provides advanced options:

```bash
# Navigate to: Actions → Advanced Firewall Configuration → Run workflow
```

**Parameters:**
- **Firewall Provider:** Choose between AWS, Azure, GCP, or Custom
- **Enable Cleanup:** Automatically remove rules after completion

## Configuration

### Required Secrets

Configure these secrets in your GitHub repository settings (`Settings → Secrets → Actions`):

#### For AWS
```
AWS_ACCESS_KEY_ID          # AWS access key
AWS_SECRET_ACCESS_KEY      # AWS secret key
AWS_REGION                 # AWS region (e.g., us-east-1)
AWS_SECURITY_GROUP_ID      # Security group ID (e.g., sg-xxxxx)
```

#### For Azure
```
AZURE_CREDENTIALS          # Azure service principal credentials (JSON)
AZURE_RESOURCE_GROUP       # Resource group name
AZURE_NSG_NAME            # Network Security Group name
```

#### For GCP
```
GCP_CREDENTIALS           # Service account key (JSON)
GCP_PROJECT              # GCP project ID
GCP_NETWORK              # VPC network name
```

#### For Custom Firewall
```
FIREWALL_API_URL         # Firewall API endpoint
FIREWALL_API_TOKEN       # API authentication token
```

### Setting Up Secrets

1. Go to repository **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Add each required secret for your firewall provider
4. Save secrets

## Implementation Details

### IP Address Validation

The system validates the detected IP address:

```bash
if [[ ! $RUNNER_IP =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
  echo "❌ Invalid IP address format: $RUNNER_IP"
  exit 1
fi
```

### Rule Naming Convention

Firewall rules use consistent naming:

```
Format: github-actions-{run_id}-{timestamp}
Example: github-actions-1234567890-1704067200
```

This ensures:
- Rules are unique per workflow run
- Easy identification and cleanup
- No conflicts with concurrent runs

### Error Handling

The workflow includes comprehensive error handling:

```yaml
- name: Configure Firewall
  continue-on-error: false  # Fail fast if firewall setup fails
  
- name: Cleanup
  if: always()  # Always cleanup, even on failure
```

## Security Considerations

### 1. Least Privilege

Only whitelist specific ports required:

```yaml
# Good: Specific port
--destination-port-ranges 443

# Avoid: All ports
--destination-port-ranges *
```

### 2. Temporary Rules

Rules are automatically removed after job completion to minimize attack surface.

### 3. IP Validation

All IP addresses are validated before creating firewall rules.

### 4. Secrets Management

Never hardcode credentials. Always use GitHub Secrets:

```yaml
# Good
env:
  API_TOKEN: ${{ secrets.FIREWALL_API_TOKEN }}

# Bad
env:
  API_TOKEN: "hardcoded-token-here"
```

### 5. Audit Trail

All firewall changes are logged in the workflow output:

```
✓ Firewall rule created for IP: 123.45.67.89
✓ Rule ID: github-actions-1234567890
✓ Firewall rules cleaned up
```

## Troubleshooting

### IP Detection Fails

**Problem:** Unable to detect runner IP address

**Solution:**
```yaml
# Use multiple IP detection services
RUNNER_IP=$(curl -s https://api.ipify.org || \
            curl -s https://ifconfig.me || \
            curl -s https://icanhazip.com)
```

### Firewall Rule Creation Fails

**Problem:** API returns error when creating rule

**Solutions:**
1. Verify credentials are correct
2. Check IAM/RBAC permissions
3. Ensure security group/NSG exists
4. Verify network connectivity

### Cleanup Doesn't Run

**Problem:** Firewall rules not removed after job

**Solution:**
```yaml
cleanup-firewall:
  if: always()  # Ensure it always runs
  needs: [configure-firewall, build-jobs]
```

### Rate Limiting

**Problem:** Too many API calls to firewall service

**Solution:**
- Cache firewall configuration
- Use job outputs to share data
- Implement exponential backoff

## Best Practices

### 1. Use Job Outputs

Share data between jobs efficiently:

```yaml
outputs:
  runner-ip: ${{ steps.get-ip.outputs.ip }}
```

### 2. Implement Idempotency

Ensure rules can be created multiple times safely:

```bash
# Check if rule exists before creating
if ! aws ec2 describe-security-group-rules --filter "Name=ip,Values=${IP}" | grep -q "${IP}"; then
  # Create rule
fi
```

### 3. Add Descriptions

Always add descriptive metadata to firewall rules:

```bash
--description "GitHub Actions - Repo: ${REPO} - Run: ${RUN_ID} - Date: ${DATE}"
```

### 4. Set Timeouts

Add timeouts to prevent hanging:

```yaml
timeout-minutes: 5
```

### 5. Monitor Usage

Track firewall rule creation/deletion:

```yaml
- name: Log Firewall Activity
  run: |
    echo "Firewall rule created at $(date)" >> firewall.log
```

## Examples

### Example 1: Access Private Package Repository

```yaml
- name: Configure Firewall for Package Repo
  run: |
    # Add rule allowing access to private npm registry
    aws ec2 authorize-security-group-ingress \
      --group-id ${{ secrets.PACKAGE_REPO_SG }} \
      --protocol tcp \
      --port 4873 \
      --cidr ${RUNNER_IP}/32

- name: Install Dependencies
  run: |
    npm config set registry https://private-npm.example.com
    npm install
```

### Example 2: Access Database for Integration Tests

```yaml
- name: Whitelist Runner for Database Access
  run: |
    # Add rule allowing PostgreSQL access
    gcloud compute firewall-rules create test-db-access-${{ github.run_id }} \
      --action=ALLOW \
      --rules=tcp:5432 \
      --source-ranges="${RUNNER_IP}/32" \
      --target-tags=database-servers

- name: Run Integration Tests
  env:
    DATABASE_URL: postgres://user:pass@db.example.com/test
  run: |
    npm run test:integration
```

### Example 3: Access Internal API

```yaml
- name: Configure Firewall for Internal API
  run: |
    curl -X POST https://api.firewall.example.com/rules \
      -H "Authorization: Bearer ${{ secrets.FIREWALL_API_TOKEN }}" \
      -d '{
        "source_ip": "'${RUNNER_IP}'",
        "destination": "internal-api.example.com",
        "port": 443,
        "protocol": "tcp",
        "action": "allow"
      }'

- name: Test API Integration
  run: |
    curl https://internal-api.example.com/health
    npm run test:api
```

## Performance Considerations

### Parallel Builds

Multiple build jobs run in parallel after firewall configuration:

```yaml
jobs:
  configure-firewall:
    # Runs first
  
  build-linux:
    needs: configure-firewall  # Waits for firewall
  
  build-macos:
    needs: configure-firewall  # Runs in parallel with build-linux
  
  build-windows:
    needs: configure-firewall  # Runs in parallel
```

### Caching

Cache dependencies to reduce build time:

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
```

## Monitoring and Alerts

### Workflow Status

Monitor workflow status in GitHub Actions:
- Green checkmark: Success
- Red X: Failure
- Yellow dot: In progress

### Notifications

Set up notifications for workflow failures:
1. Go to repository **Settings → Notifications**
2. Enable **Actions** notifications
3. Choose email or webhook

### Metrics

Track key metrics:
- Firewall configuration time
- Build success rate
- Cleanup success rate
- Average workflow duration

## Future Enhancements

Potential improvements:

1. **IP Range Caching:** Cache GitHub Actions IP ranges
2. **Multi-Region Support:** Configure firewalls in multiple regions
3. **Conditional Firewall:** Only configure when accessing protected resources
4. **Health Checks:** Verify firewall rules are working
5. **Retry Logic:** Automatic retry on transient failures
6. **Cost Optimization:** Minimize API calls to reduce costs

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS Security Groups](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html)
- [Azure NSG](https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview)
- [GCP Firewall Rules](https://cloud.google.com/vpc/docs/firewalls)
- [GitHub Actions IP Ranges](https://api.github.com/meta)

## Support

For issues or questions:

1. Check [GitHub Actions logs](../../actions)
2. Review this documentation
3. [Open an issue](../../issues/new)
4. Contact DevOps team

---

**Version:** 1.0  
**Last Updated:** 2026-01-05  
**Status:** Production Ready
