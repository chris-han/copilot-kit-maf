import { List as LucideList, LucideProps } from 'lucide-react';

const List = ({ className, ...props }: LucideProps) => {
  return <LucideList className={className} {...props} />;
};

export default List;