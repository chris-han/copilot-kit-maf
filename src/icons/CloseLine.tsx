import { X as LucideX, LucideProps } from 'lucide-react';

const CloseLine = ({ className, ...props }: LucideProps) => {
  return <LucideX className={className} {...props} />;
};

export default CloseLine;