import { CheckCircle as LucideCheckCircle, LucideProps } from 'lucide-react';

const CheckCircle = ({ className, ...props }: LucideProps) => {
  return <LucideCheckCircle className={className} {...props} />;
};

export default CheckCircle;