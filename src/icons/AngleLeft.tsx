import { ChevronLeft as LucideChevronLeft, LucideProps } from 'lucide-react';

const AngleLeft = ({ className, ...props }: LucideProps) => {
  return <LucideChevronLeft className={className} {...props} />;
};

export default AngleLeft;