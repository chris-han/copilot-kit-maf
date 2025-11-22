import { Check as LucideCheck, LucideProps } from 'lucide-react';

const CheckLine = ({ className, ...props }: LucideProps) => {
  return <LucideCheck className={className} {...props} />;
};

export default CheckLine;