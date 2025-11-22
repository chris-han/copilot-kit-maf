import { ArrowDown as LucideArrowDown, LucideProps } from 'lucide-react';

const ArrowDown = ({ className, ...props }: LucideProps) => {
  return <LucideArrowDown className={className} {...props} />;
};

export default ArrowDown;