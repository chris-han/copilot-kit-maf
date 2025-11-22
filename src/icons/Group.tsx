import { Users as LucideUsers, LucideProps } from 'lucide-react';

const Group = ({ className, ...props }: LucideProps) => {
  return <LucideUsers className={className} {...props} />;
};

export default Group;