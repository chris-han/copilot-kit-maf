import { EyeOff as LucideEyeOff, LucideProps } from 'lucide-react';

const EyeOff = ({ className, ...props }: LucideProps) => {
  return <LucideEyeOff className={className} {...props} />;
};

export default EyeOff;