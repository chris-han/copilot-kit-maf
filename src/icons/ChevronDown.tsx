import { ChevronDown as LucideChevronDown, LucideProps } from 'lucide-react';

const ChevronDown = ({ className, ...props }: LucideProps) => {
  return <LucideChevronDown className={className} {...props} />;
};

export default ChevronDown;