import { ChevronUp as LucideChevronUp, LucideProps } from 'lucide-react';

const AngleUp = ({ className, ...props }: LucideProps) => {
  return <LucideChevronUp className={className} {...props} />;
};

export default AngleUp;