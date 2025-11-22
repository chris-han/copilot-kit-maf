import { ArrowRight as LucideArrowRight, LucideProps } from 'lucide-react';

const ArrowRight = ({ className, ...props }: LucideProps) => {
  return <LucideArrowRight className={className} {...props} />;
};

export default ArrowRight;