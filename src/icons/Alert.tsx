import { AlertCircle as LucideAlertCircle, LucideProps } from 'lucide-react';

const Alert = ({ className, ...props }: LucideProps) => {
  return <LucideAlertCircle className={className} {...props} />;
};

export default Alert;