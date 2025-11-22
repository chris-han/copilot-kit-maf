import { ChevronLeft as LucideChevronLeft, LucideProps } from 'lucide-react';

const ChevronLeft = ({ className, ...props }: LucideProps) => {
  return <LucideChevronLeft className={className} {...props} />;
};

export default ChevronLeft;