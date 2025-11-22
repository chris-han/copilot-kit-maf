import { Table as LucideTable, LucideProps } from 'lucide-react';

const Table = ({ className, ...props }: LucideProps) => {
  return <LucideTable className={className} {...props} />;
};

export default Table;