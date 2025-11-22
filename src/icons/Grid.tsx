import { Grid3X3 as LucideGrid3X3, LucideProps } from 'lucide-react';

const Grid = ({ className, ...props }: LucideProps) => {
  return <LucideGrid3X3 className={className} {...props} />;
};

export default Grid;