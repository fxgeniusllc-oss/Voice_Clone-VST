# Configure Dynamic Firewall Action

A reusable GitHub Action that dynamically configures firewall rules to whitelist GitHub Actions runner IP addresses.

## Overview

This action detects the runner's public IP address and configures firewall rules to allow access to protected resources. It supports multiple cloud providers (AWS, Azure, GCP) and custom firewall APIs.

## Usage

### Basic Example

```yaml
- name: Configure Firewall
  uses: ./.github/actions/configure-firewall
  with:
    provider: 'custom'
    action: 'allow'
    port: '443'
```

### AWS Security Group Example

```yaml
- name: Allow Runner IP in AWS
  uses: ./.github/actions/configure-firewall
  with:
    provider: 'aws'
    action: 'allow'
    port: '443'
    protocol: 'tcp'
    aws-region: ${{ secrets.AWS_REGION }}
    aws-security-group-id: ${{ secrets.AWS_SECURITY_GROUP_ID }}
```

### Azure NSG Example

```yaml
- name: Configure Azure Firewall
  uses: ./.github/actions/configure-firewall
  with:
    provider: 'azure'
    action: 'allow'
    port: '5432'
    protocol: 'tcp'
    azure-resource-group: ${{ secrets.AZURE_RESOURCE_GROUP }}
    azure-nsg-name: ${{ secrets.AZURE_NSG_NAME }}
```

### GCP Firewall Example

```yaml
- name: Configure GCP Firewall
  uses: ./.github/actions/configure-firewall
  with:
    provider: 'gcp'
    action: 'allow'
    port: '3306'
    protocol: 'tcp'
    gcp-project: ${{ secrets.GCP_PROJECT }}
    gcp-network: ${{ secrets.GCP_NETWORK }}
```

### Custom API Example

```yaml
- name: Configure Custom Firewall
  uses: ./.github/actions/configure-firewall
  with:
    provider: 'custom'
    action: 'allow'
    port: '443'
    custom-api-url: ${{ secrets.FIREWALL_API_URL }}
    custom-api-token: ${{ secrets.FIREWALL_API_TOKEN }}
```

### Cleanup Example

```yaml
- name: Remove Firewall Rule
  uses: ./.github/actions/configure-firewall
  with:
    provider: 'aws'
    action: 'remove'
    port: '443'
    aws-region: ${{ secrets.AWS_REGION }}
    aws-security-group-id: ${{ secrets.AWS_SECURITY_GROUP_ID }}
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `provider` | Firewall provider: `aws`, `azure`, `gcp`, or `custom` | Yes | `custom` |
| `action` | Action to perform: `allow`, `deny`, or `remove` | Yes | `allow` |
| `port` | Port number to configure | No | `443` |
| `protocol` | Protocol: `tcp`, `udp`, or `icmp` | No | `tcp` |
| `aws-region` | AWS region (required for AWS) | No | - |
| `aws-security-group-id` | AWS Security Group ID (required for AWS) | No | - |
| `azure-resource-group` | Azure Resource Group (required for Azure) | No | - |
| `azure-nsg-name` | Azure NSG name (required for Azure) | No | - |
| `gcp-project` | GCP Project ID (required for GCP) | No | - |
| `gcp-network` | GCP VPC Network (required for GCP) | No | - |
| `custom-api-url` | Custom firewall API URL (required for custom) | No | - |
| `custom-api-token` | Custom firewall API token (required for custom) | No | - |

## Outputs

| Output | Description |
|--------|-------------|
| `runner-ip` | The detected IP address of the GitHub Actions runner |
| `rule-id` | The unique ID of the created firewall rule |
| `status` | Status of the operation: `success` or `failure` |

## Complete Workflow Example

```yaml
name: Protected Build

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      # Step 1: Configure firewall
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Allow Runner IP
        id: firewall-allow
        uses: ./.github/actions/configure-firewall
        with:
          provider: 'aws'
          action: 'allow'
          port: '443'
          aws-region: ${{ secrets.AWS_REGION }}
          aws-security-group-id: ${{ secrets.AWS_SECURITY_GROUP_ID }}
      
      # Step 2: Access protected resources
      - name: Display Firewall Info
        run: |
          echo "Runner IP: ${{ steps.firewall-allow.outputs.runner-ip }}"
          echo "Rule ID: ${{ steps.firewall-allow.outputs.rule-id }}"
          echo "Status: ${{ steps.firewall-allow.outputs.status }}"
      
      - name: Build with Access to Protected Resources
        run: |
          # Now you can access firewall-protected resources
          npm install --registry https://private-registry.example.com
          npm run build
      
      # Step 3: Cleanup (always runs)
      - name: Remove Firewall Rule
        if: always()
        uses: ./.github/actions/configure-firewall
        with:
          provider: 'aws'
          action: 'remove'
          port: '443'
          aws-region: ${{ secrets.AWS_REGION }}
          aws-security-group-id: ${{ secrets.AWS_SECURITY_GROUP_ID }}
```

## Features

- **Automatic IP Detection:** Detects runner IP using multiple services for reliability
- **IP Validation:** Validates IP address format before configuration
- **Multi-Provider Support:** Works with AWS, Azure, GCP, and custom APIs
- **Error Handling:** Fails fast on errors with clear messages
- **Unique Rule IDs:** Generates unique IDs for each workflow run
- **Cleanup Support:** Easy cleanup with `action: remove`

## Security Considerations

1. **Use GitHub Secrets:** Never hardcode credentials in workflows
2. **Least Privilege:** Only open required ports
3. **Temporary Rules:** Always cleanup rules after job completion
4. **Validate Inputs:** Action validates all inputs before execution
5. **Audit Trail:** All actions are logged in workflow output

## Troubleshooting

### IP Detection Fails

The action tries multiple IP detection services. If all fail, check:
- Runner has internet connectivity
- IP detection services are not blocked
- No proxy/VPN interfering

### Firewall Configuration Fails

Check:
- Credentials are correct and stored in GitHub Secrets
- Required permissions are granted
- Security group/NSG exists
- Network connectivity to cloud provider API

### Rule Not Removed

Ensure cleanup step has `if: always()` to run even on failure:

```yaml
- name: Cleanup
  if: always()
  uses: ./.github/actions/configure-firewall
  with:
    action: 'remove'
```

## Contributing

Improvements welcome! Please:
1. Test changes thoroughly
2. Update documentation
3. Add examples for new features

## License

MIT License - see repository LICENSE file

## Support

For issues or questions:
- Open an issue in the repository
- Check the main [FIREWALL_CONFIG.md](../FIREWALL_CONFIG.md) documentation
- Review GitHub Actions logs for detailed error messages
