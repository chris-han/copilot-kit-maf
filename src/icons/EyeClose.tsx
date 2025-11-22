import { EyeOff as LucideEyeOff, LucideProps } from 'lucide-react';

const EyeClose = ({ className, ...props }: LucideProps) => {
  return <LucideEyeOff className={className} {...props} />;
};

export default EyeClose;