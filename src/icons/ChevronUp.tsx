import { ChevronUp as LucideChevronUp, LucideProps } from 'lucide-react';

const ChevronUp = ({ className, ...props }: LucideProps) => {
  return <LucideChevronUp className={className} {...props} />;
};

export default ChevronUp;