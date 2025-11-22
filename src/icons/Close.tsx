import { X as LucideX, LucideProps } from 'lucide-react';

const Close = ({ className, ...props }: LucideProps) => {
  return <LucideX className={className} {...props} />;
};

export default Close;