import { DollarSign as LucideDollarSign, LucideProps } from 'lucide-react';

const DollarLine = ({ className, ...props }: LucideProps) => {
  return <LucideDollarSign className={className} {...props} />;
};

export default DollarLine;