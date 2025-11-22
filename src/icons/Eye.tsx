import { Eye as LucideEye, LucideProps } from 'lucide-react';

const Eye = ({ className, ...props }: LucideProps) => {
  return <LucideEye className={className} {...props} />;
};

export default Eye;