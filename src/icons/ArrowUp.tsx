import { ArrowUp as LucideArrowUp, LucideProps } from 'lucide-react';

const ArrowUp = ({ className, ...props }: LucideProps) => {
  return <LucideArrowUp className={className} {...props} />;
};

export default ArrowUp;