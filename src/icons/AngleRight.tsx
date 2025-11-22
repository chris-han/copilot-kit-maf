import { ChevronRight as LucideChevronRight, LucideProps } from 'lucide-react';

const AngleRight = ({ className, ...props }: LucideProps) => {
  return <LucideChevronRight className={className} {...props} />;
};

export default AngleRight;