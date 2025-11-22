import { ChevronDown as LucideChevronDown, LucideProps } from 'lucide-react';

const AngleDown = ({ className, ...props }: LucideProps) => {
  return <LucideChevronDown className={className} {...props} />;
};

export default AngleDown;