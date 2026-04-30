# AWS Cost Guardrails

This project should not depend on paid cloud compute. Use AWS only for short,
bounded reproducibility practice after the local, Colab, Docker, and DagsHub
workflow is working.

## Budget Rules

- Keep the AWS account on the Free plan if available.
- Create AWS Budgets alerts at `$5`, `$20`, `$50`, `$80`, and `$95`.
- Treat `$95` as the hard stop, leaving a small margin below the `$100` credit.
- Do not create EKS clusters for this project yet. EKS standard cluster support
  is billed per cluster-hour before worker-node costs.
- Avoid NAT Gateway, idle EC2 instances, orphan EBS volumes, unattached Elastic
  IPs, large S3 copies, and long-running notebooks.
- Prefer Colab for training, DagsHub for MLflow tracking, and Docker for local
  reproducibility.

## One Allowed AWS Exercise

The only recommended AWS exercise for now is a short, manually bounded run:

1. Build the Docker image locally.
2. Push or copy the image only if needed for the exercise.
3. Run one short training or inference command.
4. Log metrics to DagsHub.
5. Save any needed artifact to DagsHub or a small S3 path.
6. Terminate/delete all compute resources immediately.
7. Check Billing and Cost Management before leaving the console.

The target cost for this exercise is less than `$10`.

## What To Use If You Practice Deployment

- First choice: local Docker.
- Next choice: one short-lived EC2 instance.
- Later choice: ECS, because ECS on EC2 has no extra ECS control-plane charge
  beyond the AWS resources used.
- Defer EKS/Kubernetes until there is a real containerized service that needs
  orchestration.

## Useful References

- AWS Free Tier credits: https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html
- EC2 On-Demand billing: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html
- AWS Budgets: https://aws.amazon.com/documentation-overview/budgets/
- EKS pricing: https://aws.amazon.com/eks/pricing/
- ECS pricing: https://aws.eu/ecs/pricing/
- SageMaker managed spot training: https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
